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
from IPython.display import display, HTML
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
        self.base_dict = self.set_base_level()
        self.PDP = None 
        self.variates = {}
        self.customs = {}
        self.interactions = {}
        self.transformed_data = self.data[[self.weight] + [self.dependent]]
        self.formula = {}
        self.model = None
        self.results = None
    
    def set_base_level(self):
        self.data[self.independent + [self.weight]]
        base_dict = {}
        for item in self.independent:
            col = self.data.groupby(item)[self.weight].sum()
            base_dict[item] = col[col == max(col)].index[0]
        return base_dict
    
    def set_PDP(self):
        PDP = {}
        simple = [item for item in self.transformed_data.columns if item in self.independent]
        for item in simple:
            customs = [list(v['Z'].columns) for k,v in self.customs.items() if v['source'] == item]
            customs = [item for sublist in customs for item in sublist]
            variates = [list(v['Z'].columns) for k,v in self.variates.items() if v['source'] == item]
            variates = [item for sublist in variates for item in sublist]
            columns = [item]+customs
            columns = [val for val in [item]+customs if val in list(self.transformed_data.columns)]
            variates = [val for val in variates if val in list(self.transformed_data.columns)]
            if variates != []:
                temp = self.transformed_data.groupby(columns)[variates].mean().reset_index()
            else:
                temp = self.transformed_data.groupby(columns)['Exposure'].mean().reset_index().drop(['Exposure'], axis=1)   
            PDP[item] = temp
        tempPDP = copy.deepcopy(PDP)
        for item in tempPDP.keys():
            for append in tempPDP.keys():
                if item != append:
                    temp = tempPDP[append][tempPDP[append][append]==self.base_dict[append]]
                    for column in temp.columns:
                        PDP[item][column] = temp[column].iloc[0]
            PDP[item]['Model'] = self.results.predict(PDP[item]) 
            PDP[item].set_index(item, inplace=True)
        self.PDP = PDP
    
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
        self.variates[name] = sub_dict
        
    def create_custom(self, name, column, dictionary):
        temp = self.data[column].map(dictionary)
        temp = pd.get_dummies(temp.to_frame(), drop_first=True)
        self.customs[name] = {'source':column, 'Z':temp, 'dictionary':dictionary}
    
    def fit(self, simple=[], customs=[], variates=[], interactions=[]):
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
        self.set_formula(simple=simple, customs=customs, variates=variates, interactions=interactions)
        self.transformed_data = self.transform_data()
        self.model = sm.GLM.from_formula(formula=self.formula['formula'] , data=self.transformed_data, family=family_dict[self.family], freq_weights=self.transformed_data[self.weight])
        self.results = self.model.fit(scale=self.scale)
        fitted = self.results.predict(self.transformed_data)
        fitted.name="Fitted Avg"
        self.transformed_data = pd.concat((self.transformed_data, fitted),axis=1)
        self.set_PDP()
        
    def set_formula(self, simple=[], customs=[], variates=[], interactions=[]):
        simple_str = ' + '.join(simple)
        variate_str = ' + '.join([' + '.join(self.variates[item]['Z'].columns) for item in variates])
        custom_str = ' + '.join([' + '.join(self.customs[item]['Z'].columns) for item in customs])
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
        self.formula['formula'] = self.dependent + ' ~ ' +  simple_str + variate_str + custom_str + interaction_str    
        
    def transform_data(self, data=None):
        if data is None:
            transformed_data = self.data[self.independent + [self.weight] + [self.dependent]] 
            for i in range(len(self.formula['variates'])):
                transformed_data = pd.concat((transformed_data, self.variates[self.formula['variates'][i]]['Z']), axis=1)
            for i in range(len(self.formula['customs'])):
                transformed_data = pd.concat((transformed_data, self.customs[self.formula['customs'][i]]['Z']), axis=1)
        else:
            transformed_data = data[self.independent+[self.weight] + [self.dependent]]
            for i in range(len(self.formula['variates'])):
                name = self.formula['variates'][i]
                temp = pd.DataFrame(self.ortho_poly_predict(x=data[self.variates[name]['source']], variate=name), 
                                    columns=[name + '_p' + str(idx) for idx in range(self.variates[name]['degree'] + 1)]) 
                transformed_data = pd.concat((transformed_data, temp), axis=1)
            for i in range(len(self.formula['customs'])):
                name = self.formula['customs'][i]
                temp = data[self.customs[name]['source']].map(self.customs[name]['dictionary'])
                temp = pd.get_dummies(temp.to_frame())
                temp = temp[list(self.customs[name]['Z'].columns)]
                transformed_data = pd.concat((transformed_data, temp), axis=1)  
        return transformed_data
    
    def predict(self, data=None):
        data = self.transform_data(data)
        fitted = self.results.predict(data)
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
        
    
    def ortho_poly_fit(self, x, degree = 1, dictionary={}):
        n = degree + 1
        if dictionary != {}:
            x = x.map(dictionary) 
        x = np.asarray(x).flatten()
        if(degree >= len(np.unique(x))):
                stop("'degree' must be less than number of unique points")
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
        def view_one_way(var, fitted, model, data):
            if data is None:
                temp = pd.pivot_table(data=self.transformed_data, index=[var], values=['Loss', 'Exposure', 'Fitted Avg'], aggfunc=np.sum)
            else:
                temp = pd.pivot_table(data=self.predict(data), index=[var], values=['Loss', 'Exposure', 'Fitted Avg'], aggfunc=np.sum)
            temp['Observed'] = temp['Loss']/temp['Exposure']
            temp['Fitted'] = temp['Fitted Avg']/temp['Exposure']
            temp = temp.merge(self.PDP[var][["Model"]], how='inner', left_index=True, right_index=True)
            y_range = Range1d(start=0, end=temp['Exposure'].max()*1.8)
            if type(temp.index) == pd.core.indexes.base.Index: # Needed for categorical
                p = figure(plot_width=700, plot_height=400, y_range=y_range, title="Observed " + var, x_range=list(temp.index))
            else:
                p = figure(plot_width=700, plot_height=400, y_range=y_range, title="Observed " + var)
            # setting bar values
            h = np.array(temp['Exposure'])
            # Correcting the bottom position of the bars to be on the 0 line.
            adj_h = h/2
            # add bar renderer            
            p.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")
            # add line to secondondary axis
            p.extra_y_ranges = {"foo": Range1d(start=min(temp['Observed'].min(), temp['Model'].min())/1.1, end=max(temp['Observed'].max(), temp['Model'].max())*1.1)}
            p.add_layout(LinearAxis(y_range_name="foo"), 'right')
            # Observed Average line values
            p.line(temp.index, temp['Observed'], line_width=2, color="#ff69b4",  y_range_name="foo")
            if fitted == True:
                p.line(temp.index, temp['Fitted'], line_width=2, color="#006400", y_range_name="foo")
            if model == True:
                p.line(temp.index, temp['Model'], line_width=2, color="#00FF00", y_range_name="foo")
            show(p)
        vw = interactive(view_one_way, var=self.independent, fitted=True, model=True, data=fixed(data), layout=widgets.Layout(display="inline-flex"))
        return vw
    
    def lift_chart(self, data=None):
        if data is None:
            data = self.transformed_data
        else:
            data = self.predict(data)
        temp = data[[self.weight, self.dependent, 'Fitted Avg']].sort_values('Fitted Avg')
        temp['decile'] = (temp[self.weight].cumsum()/((sum(temp[self.weight])*1.00001)/10)+1).apply(np.floor)
        temp = pd.pivot_table(data=temp, index=['decile'], values=[self.dependent, self.weight, 'Fitted Avg'], aggfunc='sum')
        temp['Observed'] = temp[self.dependent]/temp[self.weight]
        temp['Fitted'] = temp['Fitted Avg']/temp[self.weight]
        y_range = Range1d(start=0, end=temp[self.weight].max()*1.8)
        p = figure(plot_width=700, plot_height=400, y_range=y_range, title="Lift Chart") #, x_range=list(temp.index)
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
        
        



