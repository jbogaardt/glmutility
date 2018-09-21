# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
import yaml
import copy



class GLMUtilityBase:
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
                self.base_dict[item] = self._set_base_level(data[item])
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
            self.__dict__ = self._construct_model(
                train=data,
                glm_config=self._import_model(glm_config),
                additional_fields=additional_fields,
            ).__dict__.copy()
        else:
            raise TypeError("GLM constructor not properly called.")

    def _import_model(self, file_name):
        "Loads model structure from a yaml config file"
        with open(file_name, "r") as input:
            return yaml.load(input)

    def _create_features(self, model, glm_config, feature_type):
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

    def _construct_model(self, train, glm_config, additional_fields=None):
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
        self._create_features(model, glm_config, "variates")
        self._create_features(model, glm_config, "customs")
        self._create_features(model, glm_config, "interactions")
        self._create_features(model, glm_config, "offsets")
        model.fit(
            simple=glm_config["formula"]["simple"],
            customs=glm_config["formula"]["customs"],
            variates=glm_config["formula"]["variates"],
            interactions=glm_config["formula"]["interactions"],
            offsets=glm_config["formula"]["offsets"],
        )
        return model

    def _set_base_level(self, item):
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

    def _set_PDP(self):
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
            PDP[item]["Model"] = self._link_transform(PDP[item]["Model"])
            # Creates offset PDP
            offset_subset = {
                self.offsets[item]["source"]: item for item in self.formula["offsets"]
            }
            if type(offset_subset.get(item)) is str:
                PDP[item]["Model"] = (
                    self._link_transform(
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

    def _link_transform(self, series, transform_type="linear predictor"):
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
        Z, norm2, alpha = self._ortho_poly_fit(
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
        self.base_dict[name] = self._set_base_level(temp)
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
        self._set_PDP()

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
                    self._ortho_poly_predict(
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
                temp = self._link_transform(temp)
                for i in range(len(self.formula["offsets"]) - 1):
                    offset = data[
                        self.offsets[self.formula["offsets"][i + 1]]["source"]
                    ].map(
                        self.offsets[self.formula["offsets"][i + 1]]["dictionary"]
                    )  # This works for train, but need to apply to test
                    temp = temp + self._link_transform(offset)
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
        temp = self._link_transform(temp)  # Store on linear predictor basis
        self.offsets[name] = {
            "source": source,
            "Z": temp,
            "dictionary": dictionary,
            "rescale": rescale,
        }

    def _ortho_poly_fit(self, x, degree=1, dictionary={}):
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

    def _ortho_poly_predict(self, x, variate):
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
        y_true = data[self.dependent] # Actual Loss
        y_pred = data["Fitted Avg"]/data[self.weight] # Predicted ratio
        # check and get number of samples
        temp = pd.DataFrame({'y_pred':y_pred,
                             'y_true':y_true,
                             'weight':data[self.weight]}) \
                             .sort_values('y_pred', ascending=False)
        gini_x = temp['weight'].cumsum()/temp['weight'].sum()
        gini_y = temp['y_true'].cumsum()/temp['y_true'].sum()
        gini_x = gini_x - np.append(np.array([0]),gini_x[:-1])
        gini_y = (np.append(np.array([0]),gini_y[:-1]) + gini_y)/2
        gini = np.sum(gini_x*gini_y)-.5
        return gini

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
