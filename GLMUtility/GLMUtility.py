# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file, export_png
from bokeh.layouts import gridplot
from bokeh.models import LinearAxis, Range1d, ColumnDataSource, Title
from bokeh.palettes import Spectral9
from bokeh.models import HoverTool, NumeralTickFormatter
import math
import yaml

import ipywidgets as widgets
from ipywidgets import interactive, fixed
import copy

output_notebook()


class GLM:
    """GLMUtility is built around the `statsmodels
    <http://www.statsmodels.org/devel/index.html>`_
    GLM framework. It's purpose is to create a quasi-UI in Jupyter to allow for
    a point-and-click UI experience. It is quasi as it still requires writing
    Python code, but it is heavily abstracted away.

    For the P&C actuaries out there, the workflow design is highly inspired by
    one of the more prominent GLM GUIs in the industry.

    Data prerequisite is that all variables be discrete (i.e. bin your continuous
    variables). The utility relies on the ``pandas.Dataframe``.

    Parameters:
        data : pandas.DataFrame
            A DataFrame representing the training dataset
        independent : list
            A list representing the independent or predictor variables. Each element
            of the list needs to be a column in `data` and should be discrete/binned.
        dependent : str
            A string representing the dependent or response variable for the GLM.  The
            dependent variable must exist as a column in `data`
        weight : str
            A string representing the weight of each observation for the GLM.  The
            dependent variable must exist as a column in `data`
        family : str
            The error distribution for the GLM, can be `Poisson`, `Binomial`, `Gaussian`, or `Gamma`
        link : str
            The link function for the GLM, can be `Identity`, `Log` or `Logit`
        scale : str or float
            scale can be `X2`, `dev`, or a float X2 is Pearson’s chi-square
            divided by df_resid. dev is the deviance divided by df_resid
        tweedie_var_power: float64
            If Tweedie error structure is selected, this arguement specifies the
            power of the variance function.
        glm_config : str
            Optionally, the GLM can be constructed from a .yml configuration file.
            This is ideal for when you want to reconstruct the GLM in a different
            environment.
        additional_fields: list
            A GLM constructed from a configuration file will strip all features out
            of a training set that aren't included in the GLM config. To persist additional
            features to the GLM object (e.g. primary key or diagnostic columns), add them
            the the additional_fields argument.

    Attributes:
        data : pandas.DataFrame
            A DataFrame representing the training dataset
        independent : list
            A list representing the independent or predictor variables. Each element
            of the list needs to be a column in `data` and should be discrete/binned.
        dependent : str
            A string representing the dependent or response variable for the GLM.  The
            dependent variable must exist as a column in `data`
        weight : str
            A string representing the weight of each observation for the GLM.  The
            dependent variable must exist as a column in `data`
        family : str
            The error distribution for the GLM, can be `Poisson`, `Binomial`, `Gaussian`, or `Gamma`
        link : str
            The link function for the GLM, can be `Identity`, `Log` or `Logit`
        scale : str or float
            scale can be `X2`, `dev`, or a float X2 is Pearson’s chi-square
            divided by df_resid. dev is the deviance divided by df_resid
        base_dict: dict
            The base level for each feature in `independent`, used to produce
            the partial dependence plots as well as automatically determining the
            intercept term
        PDP: dict
            For a fitted model, this represents a dictionary of the marginal plots
            for each feature in `independent`.
        variates: dict
            A dictionary of all variates in the `GLM` object
        customs: dict
            A dictionary of all customs in the `GLM` object
        interactions: dict
            A dictionary of all interactions in the `GLM` object
        offsets: dict
            A dictionary of all offsets in the `GLM` object
        transformed_data: pandas.DataFrame
            Represents `data` with all customs, variates, interactions, and offsets
            included as additional columns.
        formula: dict
            The patsy formula of the GLM as well as various other meta-data on the
            fitted model

    """

    def __init__(
        self,
        data,
        independent=None,
        dependent=None,
        weight=None,
        family="Poisson",
        link="Log",
        scale="X2",
        tweedie_var_power = 1.0,
        glm_config=None,
        additional_fields=None,
    ):
        if independent is not None and dependent is not None and weight is not None:
            self.data = data.reset_index().drop("index", axis=1)
            # make it a list of one if user only passed a single column
            independent = (
                independent if isinstance(independent, list) else [independent]
            )
            self.independent = independent
            self.dependent = dependent
            self.weight = weight
            self.scale = scale
            self.family = family
            self.link = link
            self.tweedie_var_power = tweedie_var_power
            self.base_dict = {}
            for item in self.independent:
                self.base_dict[item] = self.__set_base_level(data[item])
            self.PDP = None
            self.variates = {}
            self.customs = {}
            self.interactions = {}
            self.offsets = {}
            self.fitted_factors = {
                "simple": [],
                "customs": [],
                "variates": [],
                "interactions": [],
                "offsets": [],
            }
            self.transformed_data = self.data[[self.weight] + [self.dependent]]
            self.formula = {}
            self.comparisons = []
            self.lifts = []
            self.model = None
            self.results = None
        elif glm_config is not None:
            self.__dict__ = self.__construct_model(
                train=data,
                glm_config=self.__import_model(glm_config),
                additional_fields=additional_fields,
            ).__dict__.copy()
        else:
            raise TypeError("GLM constructor not properly called.")

    def __import_model(self, file_name):
        "Loads model structure from a yaml config file"
        with open(file_name, "r") as input:
            return yaml.load(input)

    def __create_features(self, model, glm_config, feature_type):
        "Creates features of a model from a yaml config file"
        func_dict = {
            "customs": model.create_custom,
            "variates": model.create_variate,
            "interactions": model.create_interaction,
            "offsets": model.create_offset,
        }
        for item in glm_config[feature_type].keys():
            feature = glm_config[feature_type][item]
            feature["name"] = item
            func_dict[feature_type](**feature)
        return

    def __construct_model(self, train, glm_config, additional_fields=None):
        "Constructs a model from a yaml config file"
        if additional_fields is not None:
            list_1 = glm_config["structure"]["independent"]
            list_2 = additional_fields
            main_list = list_1 + list(np.setdiff1d(list_2, list_1))
        else:
            main_list = glm_config["structure"]["independent"]
        model = GLM(
            data=train,
            independent=main_list,
            dependent=glm_config["structure"]["dependent"],
            weight=glm_config["structure"]["weight"],
            scale=glm_config["structure"]["scale"],
            family=glm_config["structure"]["family"],
            link=glm_config["structure"]["link"],
        )
        self.__create_features(model, glm_config, "variates")
        self.__create_features(model, glm_config, "customs")
        self.__create_features(model, glm_config, "interactions")
        self.__create_features(model, glm_config, "offsets")
        model.fit(
            simple=glm_config["formula"]["simple"],
            customs=glm_config["formula"]["customs"],
            variates=glm_config["formula"]["variates"],
            interactions=glm_config["formula"]["interactions"],
            offsets=glm_config["formula"]["offsets"],
        )
        return model

    def __set_base_level(self, item):
        """
        Using the specified weight to measure volume, will automatically set base level to the
        discrete level with the most volume of data.

        This gets used in the partial dependence plots, whereby the plot is set
        such that the base level prediction is the model intercept.

        It currently gets used in the model fitting for simple factors, but it needs to be recognized
        in the model fitting so that the intercept is the true intercept
        """
        data = pd.concat((item.to_frame(), self.data[self.weight].to_frame()), axis=1)
        col = data.groupby(item)[self.weight].sum()
        base_dict = col[col == max(col)].index[0]
        # Necessary for Patsy formula to recognize both str and non-str data types.
        if type(base_dict) is str:
            base_dict = "'" + base_dict + "'"
        return base_dict

    def __set_PDP(self):
        """
        This function creates a dataset for the partial dependence plots.  We identify the base
        level of each feature, and then vary the levels of the desired feature in our predictions
        """
        PDP = {}
        simple = [
            item for item in self.transformed_data.columns if item in self.independent
        ]
        for item in simple:
            customs = [
                list(v["Z"].to_frame().columns)
                for k, v in self.customs.items()
                if v["source"] == item
            ]
            customs = [item for sublist in customs for item in sublist]
            variates = [
                list(v["Z"].columns)
                for k, v in self.variates.items()
                if v["source"] == item
            ]
            variates = [item for sublist in variates for item in sublist]
            columns = [item] + customs
            columns = [
                val
                for val in [item] + customs
                if val in list(self.transformed_data.columns)
            ]
            variates = [
                val for val in variates if val in list(self.transformed_data.columns)
            ]
            if variates != []:
                temp = (
                    self.transformed_data.groupby(columns)[variates]
                    .mean()
                    .reset_index()
                )
            else:
                temp = (
                    self.transformed_data.groupby(columns)[self.weight]
                    .mean()
                    .reset_index()
                    .drop([self.weight], axis=1)
                )
            PDP[item] = temp
        tempPDP = copy.deepcopy(PDP)
        # Extract parameter table for constructing Confidence Interval Charts
        out = self.extract_params()
        intercept_se = out["CI offset"][0]
        # For every factor in [independent]
        for item in tempPDP.keys():
            # Generates the Partial Dependence Tables that will be used for making predictions and plotting
            for append in tempPDP.keys():
                if item != append:
                    if type(self.base_dict[append]) is str:
                        temp = tempPDP[append][
                            tempPDP[append][append]
                            == self.base_dict[append].replace("'", "")
                        ]
                    else:
                        temp = tempPDP[append][
                            tempPDP[append][append] == self.base_dict[append]
                        ]
                    for column in temp.columns:
                        PDP[item][column] = temp[column].iloc[0]
            # Runs predicions
            PDP[item]["Model"] = self.results.predict(PDP[item])
            PDP[item]["Model"] = self.__link_transform(PDP[item]["Model"])
            # Creates offset PDP
            offset_subset = {
                self.offsets[item]["source"]: item for item in self.formula["offsets"]
            }
            if type(offset_subset.get(item)) is str:
                PDP[item]["Model"] = (
                    self.__link_transform(
                        PDP[item][item].map(
                            self.offsets[offset_subset.get(item)]["dictionary"]
                        )
                        / self.offsets[offset_subset.get(item)]["rescale"]
                    )
                    + PDP[item]["Model"]
                )
            out_subset = out[out["field"] == item][["value", "CI offset"]]
            out_subset["value"] = out_subset["value"].astype(PDP[item][item].dtype)
            out_subset.set_index("value", inplace=True)
            PDP[item].set_index(item, inplace=True)
            PDP[item] = PDP[item].merge(
                out_subset, how="left", left_index=True, right_index=True
            )
            PDP[item]["CI offset"] = PDP[item]["CI offset"].fillna(intercept_se)
            PDP[item]["CI_U"] = PDP[item]["Model"] + PDP[item]["CI offset"]
            PDP[item]["CI_L"] = PDP[item]["Model"] - PDP[item]["CI offset"]
        self.PDP = PDP

    def __link_transform(self, series, transform_type="linear predictor"):
        "method used to toggle between linear predictor and predicted values"
        if self.link == "Log":
            if transform_type == "linear predictor":
                return np.log(series)
            else:
                return np.exp(series)
        if self.link == "Logit":
            if transform_type == "linear predictor":
                return np.log(series / (1 - series))
            else:
                return np.exp(series) / (1 + np.exp(series))
        if self.link == "Identity":
            return series

    def extract_params(self):
        """ Returns the summary statistics from the statsmodel GLM.
        """
        summary = pd.read_html(
            self.results.summary().__dict__["tables"][1].as_html(), header=0
        )[0].iloc[:, 0]
        out = pd.DataFrame()
        out["field"] = summary.str.split(",").str[0].str.replace("C\(", "")
        out["value"] = summary.str.split(".").str[1].astype(str).str.replace("]", "")
        out["param"] = self.results.__dict__["_results"].__dict__["params"]
        # Assumes normal distribution of the parameters
        out["CI offset"] = (
            self.results.__dict__["_results"].__dict__["_cache"]["bse"]
            * 1.95996398454005
        )
        return out

    def score_detail(self, data, key_column):
        """ Gets score detail for factor transparency
            DO NOT USE for anything other than Log Link
            Also not tested on Interactions"""
        source_fields = self.formula["source_fields"]
        intercept_index = (
            self.base_dict[source_fields[0]].replace("'", "")
            if type(self.base_dict[source_fields[0]]) is str
            else self.base_dict[source_fields[0]]
        )
        intercept = np.exp(self.PDP[source_fields[0]]["Model"].loc[intercept_index])

        out_data = pd.DataFrame()
        out_data[key_column] = data[key_column]
        for item in source_fields:
            out_data[item + " value"] = data[item]
            out_data[item + " model"] = np.round(
                np.exp(data[item].map(dict(self.PDP[item]["Model"]))) / intercept, 4
            )
        out_data["Total model"] = np.round(
            np.product(
                out_data[[item for item in out_data.columns if item[-5:] == "model"]],
                axis=1,
            ),
            4,
        )
        return out_data

    def create_variate(self, name, source, degree, dictionary={}):
        """method to create a variate, i.e. a polynomial smoothing of a simple factor

        Parameters:
            name : str
                A unique name for the variate being created
            source : str
                The column in `data` to which you want to apply polynomial smoothing
            degree : int
                The number of degrees you want in the polynomial
            dictionary: dict
                A mapping of the original column values to variate arguments.  The
                values of the dictionary must be numeric.

                """
        sub_dict = {}
        Z, norm2, alpha = self.__ortho_poly_fit(
            x=self.data[source], degree=degree, dictionary=dictionary
        )
        Z = pd.DataFrame(Z)
        Z.columns = [name + "_p" + str(idx) for idx in range(degree + 1)]
        sub_dict["Z"] = Z
        sub_dict["norm2"] = norm2
        sub_dict["alpha"] = alpha
        sub_dict["source"] = source
        sub_dict["degree"] = degree
        sub_dict["dictionary"] = dictionary
        self.variates[name] = sub_dict

    def create_custom(self, name, source, dictionary):
        """method to bin levels of a simple factor to a more aggregate level

        Parameters:
            name : str
                A unique name for the custom feature being created
            source : str
                The column in `data` to which you want to apply custom feature
            dictionary: dict
                A mapping of the original column values to custom binning."""
        temp = self.data[source].map(dictionary).rename(name)
        self.base_dict[name] = self.__set_base_level(temp)
        self.customs[name] = {"source": source, "Z": temp, "dictionary": dictionary}

    def fit(self, simple=[], customs=[], variates=[], interactions=[], offsets=[]):
        """Method to fit the GLM using the statsmodels packageself.

        Parameters:
            simple : list
                A list of all of the simple features you want fitted in the model.
            customs : list
                A list of all of the custom features you want fitted in the model.
            variates : list
                A list of all of the variates you want fitted in the model.
            interactions : list
                A list of all of the interactions you want fitted in the model.
            offsets : list
                A list of all of the offsets you want fitted in the model.
        """
        link_dict = {
            "Identity": sm.families.links.identity,
            "Log": sm.families.links.log,
            "Logit": sm.families.links.logit,
            "Square": sm.families.links.sqrt,
            "Probit": sm.families.links.probit,
            "Cauchy": sm.families.links.cauchy,
            "Cloglog": sm.families.links.cloglog,
            "Inverse": sm.families.links.inverse_power,
        }
        link = link_dict[self.link]
        if self.family == "Poisson":
            error_term = sm.families.Poisson(link)
        elif self.family == "Binomial":
            error_term = sm.families.Binomial(link)
        elif self.family == "Normal":
            error_term = sm.families.Gaussian(link)
        elif self.family == "Gaussian":
            error_term = sm.families.Gaussian(link)
        elif self.family == "Gamma":
            error_term = sm.families.Gamma(link)
        elif self.family == "Tweedie":
            error_term = sm.families.Tweedie(link, self.tweedie_var_power)
        self.set_formula(
            simple=simple,
            customs=customs,
            variates=variates,
            interactions=interactions,
            offsets=offsets,
        )
        self.transformed_data = self.transform_data()
        self.model = sm.GLM.from_formula(
            formula=self.formula["formula"],
            data=self.transformed_data,
            family=error_term,
            freq_weights=self.transformed_data[self.weight],
            offset=self.transformed_data["offset"],
        )
        self.results = self.model.fit(scale=self.scale)
        fitted = (
            self.results.predict(
                self.transformed_data, offset=self.transformed_data["offset"]
            )
            * self.transformed_data[self.weight]
        )
        fitted.name = "Fitted Avg"
        self.transformed_data = pd.concat((self.transformed_data, fitted), axis=1)
        self.fitted_factors = {
            "simple": simple,
            "customs": customs,
            "variates": variates,
            "interactions": interactions,
            "offsets": offsets,
        }
        self.__set_PDP()

    def set_formula(
        self, simple=[], customs=[], variates=[], interactions=[], offsets=[]
    ):
        """
        Sets the Patsy Formula for the GLM.

        Todo:
            Custom factors need a base level
        """
        simple_str = " + ".join(
            [
                "C("
                + item
                + ", Treatment(reference="
                + str(self.base_dict[item])
                + "))"
                for item in simple
            ]
        )
        variate_str = " + ".join(
            [" + ".join(self.variates[item]["Z"].columns[1:]) for item in variates]
        )
        custom_str = " + ".join(
            [
                "C("
                + item
                + ", Treatment(reference="
                + str(self.base_dict[item])
                + "))"
                for item in customs
            ]
        )
        interaction_str = " + ".join([self.interactions[item] for item in interactions])
        if simple_str != "" and variate_str != "":
            variate_str = " + " + variate_str
        if simple_str + variate_str != "" and custom_str != "":
            custom_str = " + " + custom_str
        # Only works for simple factors
        if simple_str + variate_str + custom_str != "" and interaction_str != "":
            interaction_str = " + " + interaction_str
        self.formula["simple"] = simple
        self.formula["customs"] = customs
        self.formula["variates"] = variates
        self.formula["interactions"] = interactions
        self.formula["offsets"] = offsets
        self.formula["formula"] = (
            "_response ~ " + simple_str + variate_str + custom_str + interaction_str
        )
        # Intercept only model
        if simple_str + variate_str + custom_str + interaction_str == "":
            self.formula["formula"] = self.formula["formula"] + "1"
        self.formula["source_fields"] = list(
            set(
                self.formula["simple"]
                + [self.customs[item]["source"] for item in self.formula["customs"]]
                + [self.variates[item]["source"] for item in self.formula["variates"]]
                + [self.offsets[item]["source"] for item in self.formula["offsets"]]
            )
        )

    def transform_data(self, data=None):
        """Method to add any customs, variates, interactions, and offsets to a
        generic dataset so that it can be used in the GLM object"""
        if data is None:
            # Used for training dataset
            transformed_data = self.data[
                self.independent + [self.weight] + [self.dependent]
            ]
            transformed_data = copy.deepcopy(transformed_data)
            transformed_data["_response"] = (
                transformed_data[self.dependent] / transformed_data[self.weight]
            )
            for i in range(len(self.formula["variates"])):
                transformed_data = pd.concat(
                    (transformed_data, self.variates[self.formula["variates"][i]]["Z"]),
                    axis=1,
                )
            for i in range(len(self.formula["customs"])):
                transformed_data = pd.concat(
                    (transformed_data, self.customs[self.formula["customs"][i]]["Z"]),
                    axis=1,
                )
            transformed_data["offset"] = 0
            if len(self.formula["offsets"]) > 0:
                offset = self.offsets[self.formula["offsets"][0]][
                    "Z"
                ]  # This works for train, but need to apply to test
                for i in range(len(self.formula["offsets"]) - 1):
                    offset = (
                        offset + self.offsets[self.formula["offsets"][i + 1]]["Z"]
                    )  # This works for train, but need to apply to test
                transformed_data["offset"] = offset
        else:

            transformed_data = data[
                list(
                    set(data.columns).intersection(
                        self.independent + [self.weight] + [self.dependent]
                    )
                )
            ]
            for i in range(len(self.formula["variates"])):
                name = self.formula["variates"][i]
                temp = pd.DataFrame(
                    self.__ortho_poly_predict(
                        x=data[self.variates[name]["source"]].map(
                            self.variates[name]["dictionary"]
                        ),
                        variate=name,
                    ),
                    columns=[
                        name + "_p" + str(idx)
                        for idx in range(self.variates[name]["degree"] + 1)
                    ],
                )
                transformed_data = pd.concat((transformed_data, temp), axis=1)
            for i in range(len(self.formula["customs"])):
                name = self.formula["customs"][i]
                temp = data[self.customs[name]["source"]].map(
                    self.customs[name]["dictionary"]
                )
                temp.name = name
                transformed_data = pd.concat((transformed_data, temp), axis=1)

            transformed_data["offset"] = 0
            if len(self.formula["offsets"]) > 0:
                temp = data[self.offsets[self.formula["offsets"][0]]["source"]].map(
                    self.offsets[self.formula["offsets"][0]]["dictionary"]
                )
                temp = self.__link_transform(temp)
                for i in range(len(self.formula["offsets"]) - 1):
                    offset = data[
                        self.offsets[self.formula["offsets"][i + 1]]["source"]
                    ].map(
                        self.offsets[self.formula["offsets"][i + 1]]["dictionary"]
                    )  # This works for train, but need to apply to test
                    temp = temp + self.__link_transform(offset)
                transformed_data["offset"] = temp
        return transformed_data

    def predict(self, data=None):
        """Makes predicitons off of the fitted GLM"""
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        data = self.transform_data(data)
        fitted = self.results.predict(data, offset=data["offset"])
        fitted.name = "Fitted Avg"
        return pd.concat((data, fitted), axis=1)

    # User callable
    def create_interaction(self, name, interaction):
        """ Creates an interaction term to be fit in the GLM"""
        temp = {
            **{item: "simple" for item in self.independent},
            **{item: "variate" for item in self.variates.keys()},
            **{item: "custom" for item in self.customs.keys()},
        }
        interaction_type = [temp.get(item) for item in interaction]
        transformed_interaction = copy.deepcopy(interaction)
        for i in range(len(interaction)):
            if interaction_type[i] == "variate":
                transformed_interaction[i] = list(
                    self.variates[interaction[i]]["Z"].columns
                )
            elif interaction_type[i] == "custom":
                transformed_interaction[i] = list(
                    self.customs[interaction[i]]["Z"].columns
                )
            else:
                transformed_interaction[i] = [interaction[i]]
        # Only designed to work with 2-way interaction
        self.interactions[name] = " + ".join(
            [
                val1 + ":" + val2
                for val1 in transformed_interaction[0]
                for val2 in transformed_interaction[1]
            ]
        )

    def create_offset(self, name, source, dictionary):
        """ Creates an offset term to be fit in the GLM"""
        self.data
        temp = self.data[source].map(dictionary)
        rescale = sum(self.data[self.weight] * temp) / sum(self.data[self.weight])
        temp = temp / rescale
        # This assumes that offset values are put in on real terms and not on linear predictor terms
        # We may make the choice of linear predictor and predicted value as a future argument
        temp = self.__link_transform(temp)  # Store on linear predictor basis
        self.offsets[name] = {
            "source": source,
            "Z": temp,
            "dictionary": dictionary,
            "rescale": rescale,
        }

    def __ortho_poly_fit(self, x, degree=1, dictionary={}):
        """Helper method to fit an orthogonal polynomial in the GLM.  This function
        should generally not be called by end-user and can be hidden."""
        n = degree + 1
        if dictionary != {}:
            x = x.map(dictionary)
        x = np.asarray(x).flatten()
        xbar = np.mean(x)
        x = x - xbar
        X = np.fliplr(np.vander(x, n))
        q, r = np.linalg.qr(X)
        z = np.diag(np.diag(r))
        raw = np.dot(q, z)
        norm2 = np.sum(raw ** 2, axis=0)
        alpha = (np.sum((raw ** 2) * np.reshape(x, (-1, 1)), axis=0) / norm2 + xbar)[
            :degree
        ]
        Z = raw / np.sqrt(norm2)
        return Z, norm2, alpha

    def __ortho_poly_predict(self, x, variate):
        """Helper method to make predictions off of an orthogonal polynomial in the GLM.  This function
        should generally not be called by end-user and can be hidden."""
        alpha = self.variates[variate]["alpha"]
        norm2 = self.variates[variate]["norm2"]
        degree = self.variates[variate]["degree"]
        x = np.asarray(x).flatten()
        n = degree + 1
        Z = np.empty((len(x), n))
        Z[:, 0] = 1
        if degree > 0:
            Z[:, 1] = x - alpha[0]
        if degree > 1:
            for i in np.arange(1, degree):
                Z[:, i + 1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i - 1]) * Z[
                    :, i - 1
                ]
        Z /= np.sqrt(norm2)
        return Z

    def view(self, data=None):
        """The main UI of the GLMUtility"""

        def view_one_way(var, transform, obs, fitted, model, ci, data):
            if data is None:
                temp = pd.pivot_table(
                    data=self.transformed_data,
                    index=[var],
                    values=[self.dependent, self.weight, "Fitted Avg"],
                    aggfunc=np.sum,
                )
            else:
                temp = pd.pivot_table(
                    data=self.predict(data),
                    index=[var],
                    values=[self.dependent, self.weight, "Fitted Avg"],
                    aggfunc=np.sum,
                )
            temp["Observed"] = temp[self.dependent] / temp[self.weight]
            temp["Fitted"] = temp["Fitted Avg"] / temp[self.weight]
            temp = temp.merge(
                self.PDP[var][["Model", "CI_U", "CI_L"]],
                how="inner",
                left_index=True,
                right_index=True,
            )
            if transform == "Predicted Value":
                for item in ["Model", "CI_U", "CI_L"]:
                    temp[item] = self.__link_transform(temp[item], "predicted value")
            else:
                for item in ["Observed", "Fitted"]:
                    temp[item] = self.__link_transform(temp[item], "linear predictor")
            y_range = Range1d(start=0, end=temp[self.weight].max() * 1.8)
            hover = HoverTool(
                tooltips=[("(x,y)", "($x{0.00 a}, $y{0.00 a})")], mode="mouse"
            )  # 'vline')
            if type(temp.index) == pd.core.indexes.base.Index:  # Needed for categorical
                p = figure(
                    plot_width=800,
                    y_range=y_range,
                    x_range=list(temp.index),
                    toolbar_location="right",
                    toolbar_sticky=False,
                )
            else:
                p = figure(
                    plot_width=800,
                    y_range=y_range,
                    toolbar_location="right",
                    toolbar_sticky=False,
                )

            # setting bar values
            p.add_tools(hover)
            p.add_layout(
                Title(text=var, text_font_size="12pt", align="center"), "above"
            )
            p.yaxis[0].axis_label = self.weight
            p.yaxis[0].formatter = NumeralTickFormatter(format="0.00 a")
            p.add_layout(
                LinearAxis(
                    y_range_name="foo", axis_label=self.dependent + "/" + self.weight
                ),
                "right",
            )
            h = np.array(temp[self.weight])
            # Correcting the bottom position of the bars to be on the 0 line.
            adj_h = h / 2
            # add bar renderer
            p.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")
            # add line to secondondary axis
            p.extra_y_ranges = {
                "foo": Range1d(
                    start=min(temp["Observed"].min(), temp["Model"].min()) / 1.1,
                    end=max(temp["Observed"].max(), temp["Model"].max()) * 1.1,
                )
            }
            # p.add_layout(LinearAxis(y_range_name="foo"), 'right')
            # Observed Average line values
            if obs == True:
                p.line(
                    temp.index,
                    temp["Observed"],
                    line_width=2,
                    color="#ff69b4",
                    y_range_name="foo",
                )
            if fitted == True:
                p.line(
                    temp.index,
                    temp["Fitted"],
                    line_width=2,
                    color="#006400",
                    y_range_name="foo",
                )
            if model == True:
                p.line(
                    temp.index,
                    temp["Model"],
                    line_width=2,
                    color="#00FF00",
                    y_range_name="foo",
                )
            if ci == True:
                p.line(
                    temp.index,
                    temp["CI_U"],
                    line_width=2,
                    color="#db4437",
                    y_range_name="foo",
                )
                p.line(
                    temp.index,
                    temp["CI_L"],
                    line_width=2,
                    color="#db4437",
                    y_range_name="foo",
                )
            p.xaxis.major_label_orientation = math.pi / 4
            show(p)

        var = widgets.Dropdown(
            options=self.independent, description="Field:", value=self.independent[0]
        )
        transform = widgets.ToggleButtons(
            options=["Linear Predictor", "Predicted Value"],
            button_style="",
            value="Predicted Value",
            description="Transform:",
        )
        obs = widgets.ToggleButton(
            value=True, description="Observed Value", button_style="info"
        )
        fitted = widgets.ToggleButton(
            value=True, description="Fitted Value", button_style="info"
        )
        model = widgets.ToggleButton(
            value=False, description="Model Value", button_style="warning"
        )
        ci = widgets.ToggleButton(
            value=False, description="Conf. Interval", button_style="warning"
        )
        vw = interactive(
            view_one_way,
            var=var,
            transform=transform,
            obs=obs,
            fitted=fitted,
            model=model,
            ci=ci,
            data=fixed(data),
        )
        return widgets.VBox(
            (
                widgets.HBox((var, transform)),
                widgets.HBox((obs, fitted, model, ci)),
                vw.children[-1],
            )
        )

    def lift_chart(
        self,
        data=None,
        deciles=10,
        title="",
        table=False,
        dont_show_give=False,
        file_name=None,
    ):
        """ n-Decile lift chart
        """
        if data is None:
            data = self.transformed_data
        else:
            data = self.predict(data)
            data["Fitted Avg"] = data["Fitted Avg"] * data[self.weight]
        # data = data.reset_index()
        temp = data[[self.weight, self.dependent, "Fitted Avg"]]
        temp = copy.deepcopy(temp)
        temp["sort"] = temp["Fitted Avg"] / temp[self.weight]
        temp = temp.sort_values("sort", kind="mergesort")
        # temp['decile'] = (temp[self.weight].cumsum()/((sum(temp[self.weight])*1.00001)/10)+1).apply(np.floor)
        temp["decile_initial"] = (
            temp[self.weight].cumsum()
            / ((sum(temp[self.weight]) * 1.00001) / (deciles * 2))
            + 1
        ).apply(np.floor)
        decile_map = {
            item + 1: np.ceil((item + 1) / 2 + .01) for item in range(deciles * 2)
        }
        # decile_map = {1:1,2:2,3:2,4:3,5:3,6:4,7:4,8:5,9:5,10:6,11:6,12:7,13:7,14:8,15:8,16:9,17:9,18:10,19:10,20:11}
        temp["decile"] = temp["decile_initial"].map(decile_map)
        temp = pd.pivot_table(
            data=temp,
            index=["decile"],
            values=[self.dependent, self.weight, "Fitted Avg"],
            aggfunc="sum",
        )
        temp["Observed"] = temp[self.dependent] / temp[self.weight]
        temp["Fitted"] = temp["Fitted Avg"] / temp[self.weight]
        y_range = Range1d(start=0, end=temp[self.weight].max() * 1.8)
        hover = HoverTool(
            tooltips=[("(x,y)", "($x{0.00 a}, $y{0.00 a})")], mode="mouse"
        )  # 'vline')
        p = figure(
            plot_width=600,
            plot_height=375,
            y_range=y_range,
            title="Lift Chart",
            toolbar_sticky=False,
        )  # , x_range=list(temp.index)
        p.add_tools(hover)
        h = np.array(temp[self.weight])
        # Correcting the bottom position of the bars to be on the 0 line.
        adj_h = h / 2
        # add bar renderer
        p.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")
        # add line to secondondary axis
        p.extra_y_ranges = {
            "foo": Range1d(
                start=min(temp["Observed"].min(), temp["Fitted"].min()) / 1.1,
                end=max(temp["Observed"].max(), temp["Fitted"].max()) * 1.1,
            )
        }
        if title != "":
            p.add_layout(
                Title(text=title, text_font_size="12pt", align="center"), "above"
            )
        p.add_layout(LinearAxis(y_range_name="foo"), "right")
        # Observed Average line values
        p.line(
            temp.index,
            temp["Observed"],
            line_width=2,
            color="#ff69b4",
            y_range_name="foo",
        )
        p.line(
            temp.index,
            temp["Fitted"],
            line_width=2,
            color="#006400",
            y_range_name="foo",
        )
        if table == True:
            return temp
        elif file_name is None:
            show(p)
        else:
            export_png(p, filename=file_name)

    def head_to_head(self, challenger, data=None, table=False):
        """Two way lift chart that is sorted by difference between Predicted
        scores.  Still bucketed to 10 levels with the same approximate weight
        """
        if data is None:
            data1 = self.transformed_data
            data2 = challenger.predict(self.data)
            data2["Fitted Avg"] = data2["Fitted Avg"] * data2[self.weight]
        else:
            data1 = self.predict(data)
            data1["Fitted Avg"] = data1["Fitted Avg"] * data1[self.weight]
            data2 = challenger.predict(data)
            data2["Fitted Avg"] = data2["Fitted Avg"] * data2[self.weight]
        temp = data1[[self.weight, self.dependent, "Fitted Avg"]]
        data2["Fitted Avg Challenger"] = data2["Fitted Avg"]
        data2 = data2[["Fitted Avg Challenger"]]
        temp = copy.deepcopy(temp)
        temp = temp.merge(data2, how="inner", left_index=True, right_index=True)

        temp["sort"] = temp["Fitted Avg"] / temp["Fitted Avg Challenger"]
        temp = temp.sort_values("sort")
        temp["decile"] = (
            temp[self.weight].cumsum() / ((sum(temp[self.weight]) * 1.00001) / 10) + 1
        ).apply(np.floor)
        temp = pd.pivot_table(
            data=temp,
            index=["decile"],
            values=[self.dependent, self.weight, "Fitted Avg", "Fitted Avg Challenger"],
            aggfunc="sum",
        )
        temp["Observed"] = temp[self.dependent] / temp[self.weight]
        temp["Fitted1"] = temp["Fitted Avg"] / temp[self.weight]
        temp["Fitted2"] = temp["Fitted Avg Challenger"] / temp[self.weight]
        y_range = Range1d(start=0, end=temp[self.weight].max() * 1.8)
        hover = HoverTool(
            tooltips=[("(x,y)", "($x{0.00 a}, $y{0.00 a})")], mode="mouse"
        )  # 'vline')
        p = figure(
            plot_width=700,
            plot_height=400,
            y_range=y_range,
            title="Head to Head",
            toolbar_sticky=False,
        )  # , x_range=list(temp.index)
        p.add_tools(hover)
        h = np.array(temp[self.weight])
        # Correcting the bottom position of the bars to be on the 0 line.
        adj_h = h / 2
        # add bar renderer
        p.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")
        # add line to secondondary axis
        p.extra_y_ranges = {
            "foo": Range1d(
                start=min(
                    temp["Observed"].min(), temp["Fitted1"].min(), temp["Fitted2"].min()
                )
                / 1.1,
                end=max(
                    temp["Observed"].max(), temp["Fitted1"].max(), temp["Fitted2"].max()
                )
                * 1.1,
            )
        }
        p.add_layout(LinearAxis(y_range_name="foo"), "right")
        # Observed Average line values
        p.line(
            temp.index,
            temp["Observed"],
            line_width=2,
            color="#ff69b4",
            y_range_name="foo",
        )
        p.line(
            temp.index,
            temp["Fitted1"],
            line_width=2,
            color="#006400",
            y_range_name="foo",
        )
        p.line(
            temp.index,
            temp["Fitted2"],
            line_width=2,
            color="#146195",
            y_range_name="foo",
        )
        p.legend.location = "top_left"
        if table == False:
            show(p)
        else:
            return temp

    def gini(self, data=None):
        """ This code was shamelessly lifted from `Kaggle
        <https://www.kaggle.com/jpopham91/gini-scoring-simple-and-efficient>`_
        Simple implementation of the (normalized) gini score in numpy
        Fully vectorized, no python loops, zips, etc.
        Significantly (>30x) faster than previous implementions

        """
        if data is None:
            data = self.transformed_data
        else:
            data = self.predict(data)
        # assign y_true, y_pred
        y_true = data[self.dependent]
        y_pred = data["Fitted Avg"]
        # check and get number of samples
        assert y_true.shape == y_pred.shape
        n_samples = y_true.shape[0]

        # sort rows on prediction column
        # (from largest to smallest)
        arr = np.array([y_true, y_pred]).transpose()
        true_order = arr[arr[:, 0].argsort()][::-1, 0]
        pred_order = arr[arr[:, 1].argsort()][::-1, 0]

        # get Lorenz curves
        L_true = np.cumsum(true_order) / np.sum(true_order)
        L_pred = np.cumsum(pred_order) / np.sum(pred_order)
        L_ones = np.linspace(1 / n_samples, 1, n_samples)

        # get Gini coefficients (area between curves)
        G_true = np.sum(L_ones - L_true)
        G_pred = np.sum(L_ones - L_pred)

        # normalize to true Gini coefficient
        return G_pred / G_true

    def __repr__(self):
        return self.results.summary()

    def summary(self):
        """Returns the statsmodel.api model summary"""
        return self.results.summary()

    def perfect_correlation(self):
        """ Examining correlation of factor levels
        """
        test = self.transformed_data[
            list(set(self.fitted_factors["customs"] + self.fitted_factors["simple"]))
        ]
        test2 = pd.get_dummies(test).corr()
        test3 = pd.concat(
            (
                pd.concat(
                    (
                        pd.Series(
                            np.repeat(np.array(test2.columns), test2.shape[1]),
                            name="v1",
                        ).to_frame(),
                        pd.Series(
                            np.tile(np.array(test2.columns), test2.shape[0]), name="v2"
                        ),
                    ),
                    axis=1,
                ),
                pd.Series(
                    np.triu(np.array(test2)).reshape(
                        (test2.shape[0] * test2.shape[1],)
                    ),
                    name="corr",
                ),
            ),
            axis=1,
        )
        test4 = test3[(test3["v1"] != test3["v2"]) & (test3["corr"] == 1)]
        return test4

    def two_way(self, x1, x2, pdp=False, table=False, file_name=None):
        """ Two way (two features from independent list) view of data

        """
        data = self.transformed_data
        a = (
            pd.pivot_table(
                data,
                index=x1,
                columns=x2,
                values=[self.weight, self.dependent, "Fitted Avg"],
                aggfunc="sum",
            )
            .fillna(0)
            .reset_index()
        )
        # print(a.head())
        response_list = [
            self.dependent + " " + str(item).strip() for item in (data[x2].unique())
        ]
        fitted_list = [
            "Fitted Avg " + str(item).strip() for item in (data[x2].unique())
        ]
        a.columns = [
            " ".join([str(i) for i in col]).strip() for col in a.columns.values
        ]
        a = a.fillna(0)
        a[x1] = a[x1].astype(str)
        weight_list = [
            self.weight + " " + str(item).strip() for item in data[x2].unique()
        ]
        source = ColumnDataSource(a)
        hover = HoverTool(
            tooltips=[("(x,y)", "($x{0.00 a}, $y{0.00 a})")], mode="mouse"
        )  # 'vline')
        p = figure(
            plot_width=800,
            x_range=list(a[x1]),
            toolbar_location="right",
            toolbar_sticky=False,
        )
        p.add_tools(hover)
        p.vbar_stack(
            stackers=weight_list,
            x=x1,
            source=source,
            width=0.9,
            alpha=[0.5] * len(weight_list),
            color=(Spectral9 * 100)[: len(weight_list)],
            legend=[str(item) for item in list(data[x2].unique())],
        )
        p.y_range = Range1d(0, max(np.sum(a[weight_list], axis=1)) * 1.8)
        p.xaxis[0].axis_label = x1
        p.xgrid.grid_line_color = None
        p.outline_line_color = None
        outcome = pd.DataFrame(
            np.divide(
                np.array(a[response_list]),
                np.array(a[weight_list]),
                where=np.array(a[weight_list]) > 0,
            ),
            columns=["Outcome " + str(item) for item in list(data[x2].unique())],
        )  # add line to secondondary axis
        fitted = pd.DataFrame(
            np.divide(
                np.array(a[fitted_list]),
                np.array(a[weight_list]),
                where=np.array(a[weight_list]) > 0,
            ),
            columns=["Fitted Avg " + str(item) for item in list(data[x2].unique())],
        )  # add line to secondondary axis

        p.xaxis[0].axis_label = x1
        p.yaxis[0].axis_label = self.weight
        p.yaxis[0].formatter = NumeralTickFormatter(format="0.00 a")
        p.add_layout(
            LinearAxis(
                y_range_name="foo", axis_label=self.dependent + "/" + self.weight
            ),
            "right",
        )
        p.add_layout(
            Title(text=x1 + " vs " + x2, text_font_size="12pt", align="left"), "above"
        )
        if pdp == False:
            p.extra_y_ranges = {
                "foo": Range1d(
                    start=np.min(np.array(outcome)) / 1.1,
                    end=np.max(np.array(outcome)) * 1.1,
                )
            }
            for i in range(len(outcome.columns)):
                p.line(
                    x=a[x1],
                    y=outcome.iloc[:, i],
                    line_width=3,
                    color=(Spectral9 * 100)[i],
                    line_cap="round",
                    line_alpha=.9,
                    y_range_name="foo",
                )
            for i in range(len(fitted.columns)):
                p.line(
                    x=a[x1],
                    y=fitted.iloc[:, i],
                    line_width=3,
                    color=(Spectral9 * 100)[i],
                    line_cap="round",
                    line_dash="dashed",
                    line_alpha=1,
                    y_range_name="foo",
                )
        if pdp == True:
            pdp = np.transpose(
                [
                    np.tile(self.PDP[x1].index, len(self.PDP[x2].index)),
                    np.repeat(self.PDP[x2].index, len(self.PDP[x1].index)),
                ]
            )
            pdp = pd.DataFrame(pdp, columns=[x1, x2])
            pdp = (
                (
                    (self.PDP[x1].reset_index().drop(x2, axis=1)).merge(
                        pdp, how="inner", left_on=x1, right_on=x1
                    )
                )[self.independent]
            ).to_clipboard()
            x = self.predict(pdp)
            x = pd.pivot_table(
                x, index=[x1], columns=[x2], values=["Fitted Avg"], aggfunc="mean"
            )
            p.extra_y_ranges = {
                "foo": Range1d(
                    start=np.min(np.array(x)) / 1.1, end=np.max(np.array(x)) * 1.1
                )
            }
            for i in range(len(x.columns)):
                p.line(
                    x=a[x1],
                    y=x.iloc[:, i],
                    line_width=3,
                    color=(Spectral9 * 100)[i],
                    line_cap="round",
                    line_alpha=1,
                    y_range_name="foo",
                )
        p.xaxis.major_label_orientation = math.pi / 4
        if table == True:
            return a
        elif file_name is None:
            show(p)
        else:
            export_png(p, filename=file_name)

    def create_comparisons(
        self, columns, title="", obs=True, fitted=True, model=True, ci=True, ret=False
    ):
        """This function needs to be documented..."""

        def view_one_way(transform, column, title, obs, fitted, model, ci):
            data = self.transformed_data[
                [self.dependent, self.weight, "Fitted Avg", column]
            ]
            temp = pd.pivot_table(
                data=data,
                index=[column],
                values=[self.dependent, self.weight, "Fitted Avg"],
                aggfunc=np.sum,
            )
            temp["Observed"] = temp[self.dependent] / temp[self.weight]
            temp["Fitted"] = temp["Fitted Avg"] / temp[self.weight]
            temp = temp.merge(
                self.PDP[column][["Model", "CI_U", "CI_L"]],
                how="inner",
                left_index=True,
                right_index=True,
            )
            if transform == "Predicted Value":
                for item in ["Model", "CI_U", "CI_L"]:
                    temp[item] = self.__link_transform(temp[item], "predicted value")
            else:
                for item in ["Observed", "Fitted"]:
                    temp[item] = self.__link_transform(temp[item], "linear predictor")

            y_range = Range1d(start=0, end=temp[self.weight].max() * 1.8)
            hover = HoverTool(
                tooltips=[("(x,y)", "($x{0.00 a}, $y{0.00 a})")], mode="mouse"
            )  # 'vline')
            if type(temp.index) == pd.core.indexes.base.Index:  # Needed for categorical
                f = figure(
                    plot_width=800,
                    y_range=y_range,
                    x_range=list(temp.index),
                    toolbar_location="right",
                    toolbar_sticky=False,
                )
            else:
                f = figure(
                    plot_width=800,
                    y_range=y_range,
                    toolbar_location="right",
                    toolbar_sticky=False,
                )

            # setting bar values
            f.add_tools(hover)
            f.add_layout(
                Title(text=title + column, text_font_size="12pt", align="center"),
                "above",
            )
            f.yaxis[0].axis_label = self.weight
            f.yaxis[0].formatter = NumeralTickFormatter(format="0.00 a")
            f.add_layout(
                LinearAxis(
                    y_range_name="foo", axis_label=self.dependent + "/" + self.weight
                ),
                "right",
            )
            h = np.array(temp[self.weight])

            # Correcting the bottom position of the bars to be on the 0 line.
            adj_h = h / 2

            # add bar renderer
            f.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")

            # add line to secondondary axis
            f.extra_y_ranges = {
                "foo": Range1d(
                    start=min(temp["Observed"].min(), temp["Model"].min()) / 1.1,
                    end=max(temp["Observed"].max(), temp["Model"].max()) * 1.1,
                )
            }

            # Observed Average line values
            if obs == True:
                f.line(
                    temp.index,
                    temp["Observed"],
                    line_width=2,
                    color="#ff69b4",
                    y_range_name="foo",
                )
            if fitted == True:
                f.line(
                    temp.index,
                    temp["Fitted"],
                    line_width=2,
                    color="#006400",
                    y_range_name="foo",
                )
            if model == True:
                f.line(
                    temp.index,
                    temp["Model"],
                    line_width=2,
                    color="#00FF00",
                    y_range_name="foo",
                )
            if ci == True:
                f.line(
                    temp.index,
                    temp["CI_U"],
                    line_width=2,
                    color="#db4437",
                    y_range_name="foo",
                )
                f.line(
                    temp.index,
                    temp["CI_L"],
                    line_width=2,
                    color="#db4437",
                    y_range_name="foo",
                )
            f.xaxis.major_label_orientation = math.pi / 4
            return f

        transform = "Predicted Value"
        title = title + " | " if title != "" else title
        columns = columns if isinstance(columns, list) else [columns]
        comparisons = []
        for column in columns:
            self.comparisons.append(
                view_one_way(
                    transform=transform,
                    column=column,
                    title=title,
                    obs=obs,
                    fitted=fitted,
                    model=model,
                    ci=ci,
                )
            )
            comparisons.append(self.comparisons[-1])
        if ret:
            return comparisons

    def view_comparisons(self, file_name=None, ncols=2, reorder=[]):
        """This function needs to be documented..."""

        def reorder_comparisons(order):
            if (
                max([x for x in order if x is not None]) + 1 != len(self.comparisons)
                or min([x for x in order if x is not None]) != 0
            ):
                error = (
                    """ Error, unable to reorder list because the count of reorder is not equal to the number of comparisons.
                            Use get_comparisons_count() then reorder the list that way. For instance: for a comparison count of 3
                            you may wish to do this reorder_comparisons([0,2,1]). in this case you need it to go from 0 to """
                    + str(len(self.comparisons) - 1)
                    + ". you can also add None into the list to make a blank space."
                )
                return error
            comparisons = []
            for item in order:
                if item == None:
                    comparisons.append(None)
                else:
                    comparisons.append(self.comparisons[item])
            return comparisons

        if file_name:
            if file_name[-4:] in [".png", ".PNG"]:
                export_png(self.comparisons, filename=file_name)
                return
            else:
                output_file(file_name + ".html")

        if len(self.comparisons) > 0:
            p = (
                gridplot(self.comparisons, ncols=ncols)
                if reorder == []
                else gridplot(reorder_comparisons(reorder), ncols=ncols)
            )
            # bad way of labeling the columns: can't figure out rows anway
            # toggle1 = Toggle(label=category_one, width=800)
            # toggle2 = Toggle(label=category_two, width=800)
            # show the results
            # show(layout([toggle2, toggle1], [p]))
            show(p)
        else:
            print(
                "You must create_comparisons(colums) first, before you can view_comparisons(file_name,ncols). Then you can clear_comparisons()."
            )

    def clear_comparisons(self):
        """This function needs to be documented..."""
        self.comparisons = []

    def get_comparisons(self):
        """This function needs to be documented..."""
        return self.comparisons

    def get_comparisons_count(self):
        """This function needs to be documented..."""
        return len(self.comparisons)

    def give_comparisons(self, comparison_list):
        """This function needs to be documented..."""
        for each_comparison in comparison_list:
            self.comparisons.append(each_comparison)

    def static_view(
        self,
        column,
        title,
        obs=True,
        fitted=True,
        model=True,
        ci=False,
        ret=True,
        file_name=None,
    ):
        """ This method is built off of the comparison methods.  The purpose is to
        convert the dynamic view() method into a programmatic expression for quick file
        output."""
        temp = list(self.comparisons)
        self.clear_comparisons()
        comparisons = self.create_comparisons(
            [column], title=title, obs=obs, fitted=fitted, model=model, ci=ci, ret=ret
        )
        self.view_comparisons(file_name)
        self.comparisons = list(temp)

    def create_lift(self, data=None, title="", ret=False):
        """This function needs to be documented..."""
        if data is not None:
            data = data.reset_index()
        lift = self.lift_chart(data=None, title=title, dont_show_give=True)
        self.lifts.append(lift)
        if ret:
            return [lift]

    def view_lifts(self, file_name=None, ncols=2, reorder=[]):
        """This function needs to be documented..."""

        def reorder_lifts(order):
            if (
                max([x for x in order if x is not None]) + 1 != len(self.lifts)
                or min([x for x in order if x is not None]) != 0
            ):
                error = (
                    """ Error, unable to reorder list because the count of reorder is not equal to the number of comparisons.
                            Use get_comparisons_count() then reorder the list that way. For instance: for a comparison count of 3
                            you may wish to do this reorder_comparisons([0,2,1]). in this case you need it to go from 0 to """
                    + str(len(self.lifts) - 1)
                    + ". you can also add None into the list to make a blank space."
                )
                return error
            lifts = []
            for item in order:
                if item == None:
                    lifts.append(None)
                else:
                    lifts.append(self.lifts[item])
            return lifts

        if file_name:
            output_file(file_name + ".html")

        if len(self.lifts) > 0:
            p = (
                gridplot(self.lifts, ncols=ncols)
                if reorder == []
                else gridplot(reorder_lifts(reorder), ncols=ncols)
            )
            # bad way of labeling the columns: can't figure out rows anway
            # toggle1 = Toggle(label=category_one, width=800)
            # toggle2 = Toggle(label=category_two, width=800)
            # show the results
            # show(layout([toggle2, toggle1], [p]))
            show(p)
        else:
            print(
                "You must create_lift() first, before you can view_lifts(file_name,ncols). Then you can clear_lifts()."
            )

    def clear_lifts(self):
        """This function needs to be documented..."""
        self.lifts = []

    def get_lifts(self):
        """This function needs to be documented..."""
        return self.lifts

    def get_lifts_count(self):
        """This function needs to be documented..."""
        return len(self.lifts)

    def give_lifts(self, lift_list):
        """This function needs to be documented..."""
        for each_lift in lift_list:
            self.lifts.append(each_lift)

    def export_model(self, file_name):
        """  This method exports a YAML file with all configurations of a GLM model.
        This makes it easy to add the GLM to a version control system like git as well
        as rebuild the model in a different environment.

        Arguments:
            file_name: str
            This is the filepath and name to export the model yaml file to.

        Returns:
            None

        """
        v = {
            item: {
                k: self.variates[item][k] for k in ["source", "degree", "dictionary"]
            }
            for item in self.variates.keys()
        }
        c = {
            item: {k: self.customs[item][k] for k in ["source", "dictionary"]}
            for item in self.customs.keys()
        }
        o = {
            item: {k: self.offsets[item][k] for k in ["source", "dictionary"]}
            for item in self.offsets.keys()
        }
        x = {
            item: list(np.sort(self.data[item].unique()))
            for item in self.formula["simple"]
        }
        i = self.interactions
        f = self.fitted_factors
        s = {
            "weight": self.weight,
            "dependent": self.dependent,
            "independent": self.formula["source_fields"],
            "family": self.family,
            "link": self.link,
            "scale": self.scale,
        }
        with open(file_name, "w") as outfile:
            yaml.dump(
                {
                    "simple": x,
                    "variates": v,
                    "customs": c,
                    "offsets": o,
                    "formula": f,
                    "interactions": i,
                    "structure": s,
                },
                outfile,
                default_flow_style=False,
            )
