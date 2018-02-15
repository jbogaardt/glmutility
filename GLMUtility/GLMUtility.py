# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:16:50 2017

@author: jboga
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.models import LinearAxis, Range1d
import ipywidgets as widgets
from ipywidgets import interactive, fixed
import copy
output_notebook()

class GLM():
    def __init__(self, data, independent, dependent, weight, family='Poisson', link='Log', scale='X2'):
        self.data = data
        self.independent = independent
        self.dependent = dependent
        self.weight = weight
        self.scale = scale
        self.family = family
        self.link = link
        self.base_dict = {}
        #self.base_dict = self.set_base_level()
        for item in self.independent:
            self.base_dict[item] = self.set_base_level(data[item])
        self.PDP = None
        self.variates = {}
        self.customs = {}
        self.interactions = {}
        self.offsets = {}
        self.fitted_factors = {'simple':[], 'customs':[], 'variates':[],'interactions':[],'offsets':[]}
        self.transformed_data = self.data[[self.weight] + [self.dependent]]
        self.formula = {}
        self.model = None
        self.results = None
<<<<<<< HEAD

    def set_base_level(self, item):
        """
        Using the specified weight to measure volume, will automatically set base level to the
        discrete level with the most volume of data.

        This gets used in the partial dependence plots, whereby the plot is set
        such that the base level prediction is the model intercept.

        It currently gets used in the model fitting for simple factors, but it needs to be recognized
        in the model fitting so that the intercept is the true intercept
        """
        data =  pd.concat((item.to_frame(), self.data[self.weight].to_frame()),axis=1)
        col = data.groupby(item)[self.weight].sum()
        base_dict = col[col == max(col)].index[0]
        # Necessary for Patsy formula to recognize both str and non-str data types.
        if type(base_dict) is str:
            base_dict = '\'' + base_dict + '\''
=======
    
    def set_base_level(self):
        """
        Using the specified weight to measure volume, will automatically set base level to the
        discrete level with the most volume of data.
        
        This gets used in the partial dependence plots, whereby the plot is set
        such that the base level prediction is the model intercept.
        
        It currently gets used in the model fitting for simple factors, but it needs to be recognized
        in the model fitting so that the intercept is the true intercept
        """
        self.data[self.independent + [self.weight]]
        base_dict = {}
        for item in self.independent:
            col = self.data.groupby(item)[self.weight].sum()
            base_dict[item] = col[col == max(col)].index[0]
            # Necessary for Patsy formula to recognize both str and non-str data types.
            if type(base_dict[item]) is str:
                base_dict[item] = '\'' + base_dict[item] + '\''
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
        return base_dict

    def set_PDP(self):
        PDP = {}
        simple = [item for item in self.transformed_data.columns if item in self.independent]
        for item in simple:
            customs = [list(v['Z'].to_frame().columns) for k,v in self.customs.items() if v['source'] == item]
            customs = [item for sublist in customs for item in sublist]
            variates = [list(v['Z'].columns) for k,v in self.variates.items() if v['source'] == item]
            variates = [item for sublist in variates for item in sublist]
            columns = [item]+customs
            columns = [val for val in [item]+customs if val in list(self.transformed_data.columns)]
            variates = [val for val in variates if val in list(self.transformed_data.columns)]
            if variates != []:
                temp = self.transformed_data.groupby(columns)[variates].mean().reset_index()
            else:
<<<<<<< HEAD
                temp = self.transformed_data.groupby(columns)[self.weight].mean().reset_index().drop([self.weight], axis=1)
=======
                temp = self.transformed_data.groupby(columns)[self.weight].mean().reset_index().drop([self.weight], axis=1)   
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
            PDP[item] = temp
        tempPDP = copy.deepcopy(PDP)
        # Extract parameter table for constructing Confidence Interval Charts
        out = self.extract_params()
        intercept_se = out['CI offset'][0]
        # For every factor in [independent]
        for item in tempPDP.keys():
            # Generates the Partial Dependence Tables that will be used for making predictions and plotting
            for append in tempPDP.keys():
                if item != append:
                    if type(self.base_dict[append]) is str:
                        temp = tempPDP[append][tempPDP[append][append]==self.base_dict[append].replace('\'','')]
                    else:
                        temp = tempPDP[append][tempPDP[append][append]==self.base_dict[append]]
                    for column in temp.columns:
                        PDP[item][column] = temp[column].iloc[0]
<<<<<<< HEAD
            #Runs predicions
            PDP[item]['Model'] = self.results.predict(PDP[item])
            PDP[item]['Model'] = self.link_transform(PDP[item]['Model'])
            # Creates offset PDP
            offset_subset = {self.offsets[item]['source']:item for item in self.formula['offsets']}
            if type(offset_subset.get(item)) is str:
                PDP[item]['Model'] =  self.link_transform(PDP[item][item].map(self.offsets[offset_subset.get(item)]['dictionary'])/self.offsets[offset_subset.get(item)]['rescale']) + PDP[item]['Model']
=======
            PDP[item]['Model'] = self.results.predict(PDP[item])
            PDP[item]['Model'] = self.link_transform(PDP[item]['Model'])
            offset_subset = {v['source']:k for k,v in self.offsets.items()} 
            if type(offset_subset.get(item)) is str:
                PDP[item]['Model'] =  PDP[item]['Model'] + PDP[item][item].map(self.offsets[offset_subset.get(item)]['dictionary'])
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
            out_subset = out[out['field']==item][['value', 'CI offset']]
            out_subset['value'] = out_subset['value'].astype(PDP[item][item].dtype)
            out_subset.set_index('value',inplace=True)
            PDP[item].set_index(item, inplace=True)
            PDP[item] = PDP[item].merge(out_subset,how='left', left_index=True, right_index=True)
            PDP[item]['CI offset'] = PDP[item]['CI offset'].fillna(intercept_se)
            PDP[item]['CI_U'] = PDP[item]['Model'] + PDP[item]['CI offset']
<<<<<<< HEAD
            PDP[item]['CI_L'] = PDP[item]['Model'] - PDP[item]['CI offset']
        self.PDP = PDP

=======
            PDP[item]['CI_L'] = PDP[item]['Model'] - PDP[item]['CI offset']          
        self.PDP = PDP
    
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
    def link_transform(self, series, transform_type='linear predictor'):
        if self.link == 'Log':
            if transform_type == 'linear predictor':
                return np.log(series)
            else:
                return np.exp(series)
        if self.link == 'Logit':
            if transform_type == 'linear predictor':
                return np.log(series/(1-series))
            else:
                return np.exp(series)/(1 + np.exp(series))
        if self.link == 'Identity':
            return series
<<<<<<< HEAD

=======
        
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
    def extract_params(self):
        summary = pd.read_html(self.results.summary().__dict__['tables'][1].as_html(), header=0)[0].iloc[:,0]
        out = pd.DataFrame()
        out['field'] = summary.str.split(",").str[0].str.replace('C\(','')
        out['value'] = summary.str.split(".").str[1].astype(str).str.replace(']','')
        out['param'] = self.results.__dict__['_results'].__dict__['params']
        out['CI offset'] = self.results.__dict__['_results'].__dict__['_cache']['bse']*1.95996398454005
        return out
<<<<<<< HEAD

=======
        
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
    def create_variate(self, name, column, degree, dictionary={}):
        sub_dict = {}
        Z, norm2, alpha = self.ortho_poly_fit(x = self.data[column], degree=degree, dictionary=dictionary)
        Z = pd.DataFrame(Z)
        Z.columns = [name + '_p' + str(idx) for idx in range(degree + 1)]
        sub_dict['Z'] = Z
        sub_dict['norm2'] = norm2
        sub_dict['alpha'] = alpha
        sub_dict['source'] = column
        sub_dict['degree'] = degree
        sub_dict['dictionary'] = dictionary
        self.variates[name] = sub_dict

    def create_custom(self, name, column, dictionary):
        temp = self.data[column].map(dictionary).rename(name)
        self.base_dict[name] = self.set_base_level(temp)
        #temp = pd.get_dummies(temp.to_frame(), drop_first=True)
        self.customs[name] = {'source':column, 'Z':temp, 'dictionary':dictionary}
<<<<<<< HEAD

=======
    
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
    def fit(self, simple=[], customs=[], variates=[], interactions=[], offsets=[]):
        link_dict = {'Identity':sm.families.links.identity,
                     'Log':sm.families.links.log,
                     'Logit':sm.families.links.logit}
        link = link_dict[self.link]
        family_dict = {'Poisson':sm.families.Poisson(link),
                       'Binomial':sm.families.Binomial(link),
                       'Normal':sm.families.Gaussian(link),
                       'Gaussian':sm.families.Gaussian(link),
                       'Gamma':sm.families.Gamma(link)
                       }
<<<<<<< HEAD
        self.set_formula(simple=simple, customs=customs, variates=variates, interactions=interactions, offsets=offsets)
        self.transformed_data = self.transform_data()
        self.model = sm.GLM.from_formula(formula=self.formula['formula'] , data=self.transformed_data, family=family_dict[self.family], freq_weights=self.transformed_data[self.weight], offset=self.transformed_data['offset'])
        self.results = self.model.fit(scale=self.scale)
        fitted = self.results.predict(self.transformed_data, offset=self.transformed_data['offset'])*self.transformed_data[self.weight]
        fitted.name="Fitted Avg"
        self.transformed_data = pd.concat((self.transformed_data, fitted),axis=1)
        self.fitted_factors = {'simple':simple, 'customs':customs,'variates':variates, 'interactions':interactions,'offsets':offsets}
        self.set_PDP()

    def set_formula(self, simple=[], customs=[], variates=[], interactions=[], offsets=[]):
        '''
        Sets the Patsy Formula for the GLM.

=======
        self.set_formula(simple=simple, customs=customs, variates=variates, interactions=interactions)
        if len(offsets) > 0:
            offset = self.offsets[offsets[0]]['Z']
        self.transformed_data = self.transform_data()
        self.model = sm.GLM.from_formula(formula=self.formula['formula'] , data=self.transformed_data, family=family_dict[self.family], freq_weights=self.transformed_data[self.weight], offset=offset)
        self.results = self.model.fit(scale=self.scale)
        fitted = self.results.predict(self.transformed_data, offset=offset)*self.transformed_data[self.weight]
        fitted.name="Fitted Avg"
        self.transformed_data = pd.concat((self.transformed_data, fitted),axis=1)
        self.fitted_factors = {'simple':simple, 'customs':customs,'variates':variates, 'interactions':interactions,'offsets':offsets} 
        self.set_PDP()
        
    def set_formula(self, simple=[], customs=[], variates=[], interactions=[]):
        '''
        Sets the Patsy Formula for the GLM.
        
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
        Todo:
            Custom factors need a base level
        '''
        #simple_str = ' + '.join(simple)
        simple_str = ' + '.join(['C(' + item + ', Treatment(reference=' + str(self.base_dict[item]) + '))' for item in simple])
        variate_str = ' + '.join([' + '.join(self.variates[item]['Z'].columns[1:]) for item in variates])
<<<<<<< HEAD
        custom_str = ' + '.join(['C(' + item + ', Treatment(reference=' + str(self.base_dict[item]) + '))' for item in customs])
        #custom_str = ' + '.join([' + '.join(self.customs[item]['Z'].columns) for item in customs])
=======
        custom_str = ' + '.join([' + '.join(self.customs[item]['Z'].columns) for item in customs])
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
        interaction_str = ' + '.join([self.interactions[item] for item in interactions])
        if simple_str != '' and variate_str != '':
            variate_str = ' + ' + variate_str
        if simple_str + variate_str != '' and custom_str != '':
            custom_str = ' + ' + custom_str
        # Only works for simple factors
        if simple_str + variate_str + custom_str != '' and interaction_str != '':
            interaction_str = ' + ' + interaction_str
        self.formula['simple'] = simple
        self.formula['customs'] = customs
        self.formula['variates'] = variates
        self.formula['interactions'] = interactions
<<<<<<< HEAD
        self.formula['offsets'] = offsets
        #self.formula['formula'] = self.dependent + ' ~ ' +  simple_str + variate_str + custom_str + interaction_str
        self.formula['formula'] = '_response ~ ' +  simple_str + variate_str + custom_str + interaction_str
=======
        #self.formula['formula'] = self.dependent + ' ~ ' +  simple_str + variate_str + custom_str + interaction_str    
        self.formula['formula'] = '_response ~ ' +  simple_str + variate_str + custom_str + interaction_str    
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
        # Intercept only model
        if simple_str + variate_str + custom_str + interaction_str == '':
            self.formula['formula'] = self.formula['formula'] + '1'

    def transform_data(self, data=None):
        if data is None:
<<<<<<< HEAD
            # Used for training dataset
            transformed_data = self.data[self.independent + [self.weight] + [self.dependent]]
=======
            transformed_data = self.data[self.independent + [self.weight] + [self.dependent]] 
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
            transformed_data = copy.deepcopy(transformed_data)
            transformed_data['_response'] = transformed_data[self.dependent] / transformed_data[self.weight]
            for i in range(len(self.formula['variates'])):
                transformed_data = pd.concat((transformed_data, self.variates[self.formula['variates'][i]]['Z']), axis=1)
            for i in range(len(self.formula['customs'])):
                transformed_data = pd.concat((transformed_data, self.customs[self.formula['customs'][i]]['Z']), axis=1)
            transformed_data['offset'] = 0
            if len(self.formula['offsets']) > 0:
                offset = self.offsets[self.formula['offsets'][0]]['Z'] # This works for train, but need to apply to test
                for i in range(len(self.formula['offsets'])-1):
                    offset = offset + self.offsets[self.formula['offsets'][i + 1]]['Z'] # This works for train, but need to apply to test
                transformed_data['offset'] = offset
        else:
            # Used for new dataset
            #transformed_data = data[self.independent+[self.weight] + [self.dependent]]
            transformed_data = data[list(set(data.columns).intersection(self.independent+[self.weight] + [self.dependent]))]
            for i in range(len(self.formula['variates'])):
                name = self.formula['variates'][i]
                temp = pd.DataFrame(self.ortho_poly_predict(x=data[self.variates[name]['source']], variate=name),
                                    columns=[name + '_p' + str(idx) for idx in range(self.variates[name]['degree'] + 1)])
                transformed_data = pd.concat((transformed_data, temp), axis=1)
            for i in range(len(self.formula['customs'])):
                name = self.formula['customs'][i]
                temp = data[self.customs[name]['source']].map(self.customs[name]['dictionary'])
<<<<<<< HEAD
                temp.name = name
                #temp = pd.get_dummies(temp.to_frame())
                #temp = temp[list(self.customs[name]['Z'].columns)] # What does this even do?
                transformed_data = pd.concat((transformed_data, temp), axis=1)
            # Create offsets in data transformation for new datasets?
            transformed_data['offset'] = 0
            if len(self.formula['offsets']) > 0:
                temp = data[self.offsets[self.formula['offsets'][0]]['source']].map(self.offsets[self.formula['offsets'][0]]['dictionary'])
                temp = self.link_transform(temp)
                #offset = self.offsets[self.formula['offsets'][0]]['Z'] # This works for train, but need to apply to test
                for i in range(len(self.formula['offsets'])-1):
                    offset = data[self.offsets[self.formula['offsets'][i+1]]['source']].map(self.offsets[self.formula['offsets'][i+1]]['dictionary'])# This works for train, but need to apply to test
                    temp = temp + self.link_transform(offset)
                transformed_data['offset'] = temp
=======
                temp = pd.get_dummies(temp.to_frame())
                temp = temp[list(self.customs[name]['Z'].columns)]
                transformed_data = pd.concat((transformed_data, temp), axis=1) 
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
        return transformed_data

    #def compute_offset(self):


    def predict(self, data=None):
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        data = self.transform_data(data)
        fitted = self.results.predict(data, offset=data['offset'])*data[self.weight]
        #fitted = self.results.predict(data)
        fitted.name="Fitted Avg"
        return pd.concat((data, fitted),axis=1)

    # User callable
    def create_interaction(self, name, interaction):
        temp = {**{item:'simple' for item in self.independent},
                **{item:'variate' for item in self.variates.keys()},
                **{item:'custom' for item in self.customs.keys()}}
        interaction_type = [temp.get(item) for item in interaction]
        transformed_interaction = copy.deepcopy(interaction)
        for i in range(len(interaction)):
            if interaction_type[i] == 'variate':
                transformed_interaction[i] = list(self.variates[interaction[i]]['Z'].columns)
            elif interaction_type[i] == 'custom':
                transformed_interaction[i] = list(self.customs[interaction[i]]['Z'].columns)
            else:
                transformed_interaction[i] = [interaction[i]]
        # Only designed to work with 2-way interaction
        self.interactions[name] =  ' + '.join([val1+':'+val2 for val1 in transformed_interaction[0] for val2 in transformed_interaction[1]])
<<<<<<< HEAD

    def create_offset(self, name, column, dictionary):
        self.data
        temp = self.data[column].map(dictionary)
        rescale = sum(self.data[self.weight]*temp)/sum(self.data[self.weight])
        temp = temp/rescale
        # This assumes that offset values are put in on real terms and not on linear predictor terms
        # We may make the choice of linear predictor and predicted value as a future argument
        temp = self.link_transform(temp) # Store on linear predictor basis
        self.offsets[name] = {'source':column, 'Z':temp, 'dictionary':dictionary, 'rescale':rescale}

=======
        
    def create_offset(self, name, column, dictionary):
        temp = self.data[column].map(dictionary)
        temp = self.link_transform(temp)
        self.offsets[name] = {'source':column, 'Z':temp, 'dictionary':dictionary}
        
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
    def ortho_poly_fit(self, x, degree = 1, dictionary={}):
        n = degree + 1
        if dictionary != {}:
            x = x.map(dictionary)
        x = np.asarray(x).flatten()
        xbar = np.mean(x)
        x = x - xbar
        X = np.fliplr(np.vander(x, n))
        q,r = np.linalg.qr(X)
        z = np.diag(np.diag(r))
        raw = np.dot(q, z)
        norm2 = np.sum(raw**2, axis=0)
        alpha = (np.sum((raw**2)*np.reshape(x,(-1,1)), axis=0)/norm2 + xbar)[:degree]
        Z = raw / np.sqrt(norm2)
        return Z, norm2, alpha

    def ortho_poly_predict(self, x, variate):
        alpha = self.variates[variate]['alpha']
        norm2 = self.variates[variate]['norm2']
        degree = self.variates[variate]['degree']
        dictionary = self.variates[variate]['dictionary']
        x = x.map(dictionary)
        x = np.asarray(x).flatten()
        n = degree + 1
        Z = np.empty((len(x), n))
        Z[:,0] = 1
        if degree > 0:
            Z[:, 1] = x - alpha[0]
        if degree > 1:
          for i in np.arange(1,degree):
              Z[:, i+1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
        Z /= np.sqrt(norm2)
        return Z

    def view(self, data = None):
        def view_one_way(var, transform, obs, fitted, model, ci, data):
            if data is None:
                temp = pd.pivot_table(data=self.transformed_data, index=[var], values=[self.dependent, self.weight, 'Fitted Avg'], aggfunc=np.sum)
<<<<<<< HEAD
            else:
                temp = pd.pivot_table(data=self.predict(data), index=[var], values=[self.dependent, self.weight, 'Fitted Avg'], aggfunc=np.sum)
            temp['Observed'] = temp[self.dependent]/temp[self.weight]
            temp['Fitted'] = temp['Fitted Avg']/temp[self.weight]
            temp = temp.merge(self.PDP[var][['Model','CI_U','CI_L']], how='inner', left_index=True, right_index=True)
            if transform == 'Predicted Value':
                for item in ['Model','CI_U','CI_L']:
                    temp[item] = self.link_transform(temp[item],'predicted value')
            else:
=======
            else:
                temp = pd.pivot_table(data=self.predict(data), index=[var], values=[self.dependent, self.weight, 'Fitted Avg'], aggfunc=np.sum)
            temp['Observed'] = temp[self.dependent]/temp[self.weight]
            temp['Fitted'] = temp['Fitted Avg']/temp[self.weight]
            temp = temp.merge(self.PDP[var][['Model','CI_U','CI_L']], how='inner', left_index=True, right_index=True)
            if transform == 'Predicted Value':
                for item in ['Model','CI_U','CI_L']:
                    temp[item] = self.link_transform(temp[item],'predicted value')
            else:
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
                for item in ['Observed','Fitted']:
                    temp[item] = self.link_transform(temp[item],'linear predictor')
            y_range = Range1d(start=0, end=temp[self.weight].max()*1.8)
            if type(temp.index) == pd.core.indexes.base.Index: # Needed for categorical
                p = figure(plot_width=700, plot_height=400, y_range=y_range, title=var, x_range=list(temp.index), toolbar_location = 'right', toolbar_sticky=False)
            else:
                p = figure(plot_width=700, plot_height=400, y_range=y_range, title=var, toolbar_location = 'right', toolbar_sticky=False)
            # setting bar values
            h = np.array(temp[self.weight])
            # Correcting the bottom position of the bars to be on the 0 line.
            adj_h = h/2
            # add bar renderer
            p.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")
            # add line to secondondary axis
            p.extra_y_ranges = {"foo": Range1d(start=min(temp['Observed'].min(), temp['Model'].min())/1.1, end=max(temp['Observed'].max(), temp['Model'].max())*1.1)}
            p.add_layout(LinearAxis(y_range_name="foo"), 'right')
            # Observed Average line values
            if obs == True:
                p.line(temp.index, temp['Observed'], line_width=2, color="#ff69b4",  y_range_name="foo")
            if fitted == True:
                p.line(temp.index, temp['Fitted'], line_width=2, color="#006400", y_range_name="foo")
            if model == True:
                p.line(temp.index, temp['Model'], line_width=2, color="#00FF00", y_range_name="foo")
            if ci == True:
                p.line(temp.index, temp['CI_U'], line_width=2, color="#db4437", y_range_name="foo")
                p.line(temp.index, temp['CI_L'], line_width=2, color="#db4437", y_range_name="foo")
<<<<<<< HEAD

=======
                
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
            show(p)
        var = widgets.Dropdown(options=self.independent, description='Field:', value=self.independent[0])
        transform = widgets.ToggleButtons(options=['Linear Predictor', 'Predicted Value'],button_style='', value='Predicted Value',description="Transform:")
        obs = widgets.ToggleButton(value=True,description='Observed Value',button_style='info')
        fitted = widgets.ToggleButton(value=True,description='Fitted Value',button_style='info')
        model = widgets.ToggleButton(value=False,description='Model Value',button_style='warning')
        ci = widgets.ToggleButton(value=False,description='Conf. Interval',button_style='warning')
        vw = interactive(view_one_way, var=var, transform=transform, obs=obs, fitted=fitted, model=model, ci=ci, data=fixed(data))
        return widgets.VBox((widgets.HBox((var,transform)), widgets.HBox((obs, fitted,model, ci)),vw.children[-1]))
<<<<<<< HEAD


=======
    
    
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e
    def lift_chart(self, data=None):
        if data is None:
            data = self.transformed_data
        else:
            data = self.predict(data)
        temp = data[[self.weight, self.dependent, 'Fitted Avg']]
        temp = copy.deepcopy(temp)
        temp['sort'] = temp['Fitted Avg']/temp[self.weight]
        temp = temp.sort_values('sort')
        temp['decile'] = (temp[self.weight].cumsum()/((sum(temp[self.weight])*1.00001)/10)+1).apply(np.floor)
        temp = pd.pivot_table(data=temp, index=['decile'], values=[self.dependent, self.weight, 'Fitted Avg'], aggfunc='sum')
        temp['Observed'] = temp[self.dependent]/temp[self.weight]
        temp['Fitted'] = temp['Fitted Avg']/temp[self.weight]
        y_range = Range1d(start=0, end=temp[self.weight].max()*1.8)
        p = figure(plot_width=700, plot_height=400, y_range=y_range, title="Lift Chart", toolbar_sticky=False) #, x_range=list(temp.index)
        h = np.array(temp[self.weight])
        # Correcting the bottom position of the bars to be on the 0 line.
        adj_h = h/2
        # add bar renderer
        p.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")
        # add line to secondondary axis
        p.extra_y_ranges = {"foo": Range1d(start=min(temp['Observed'].min(), temp['Fitted'].min())/1.1, end=max(temp['Observed'].max(), temp['Fitted'].max())*1.1)}
        p.add_layout(LinearAxis(y_range_name="foo"), 'right')
        # Observed Average line values
        p.line(temp.index, temp['Observed'], line_width=2, color="#ff69b4",  y_range_name="foo")
        p.line(temp.index, temp['Fitted'], line_width=2, color="#006400", y_range_name="foo")
        show(p)
<<<<<<< HEAD
=======
    
    def summary(self):
        return self.results.summary()
        
        
>>>>>>> 8ee5180b936619b27fb76bc173c9091668063d0e

    def __repr__(self):
        return self.results.summary()

    def summary(self):
        return self.results.summary()

    def perfect_correlation(self):
        # Examining correlation of factor levels
        test = self.transformed_data[list(set(self.fitted_factors['customs']+self.fitted_factors['simple']))]
        test2 = pd.get_dummies(test).corr()
        test3 = pd.concat((pd.concat((pd.Series(np.repeat(np.array(test2.columns),test2.shape[1]),name='v1').to_frame(),
            pd.Series(np.tile(np.array(test2.columns),test2.shape[0]),name='v2')),axis=1),
            pd.Series(np.triu(np.array(test2)).reshape((test2.shape[0]*test2.shape[1],)),name='corr')),axis=1)
        test4 = test3[(test3['v1']!=test3['v2']) & (test3['corr'] == 1)]
        return test4
