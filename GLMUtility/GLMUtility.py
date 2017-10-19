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
from ipywidgets import interactive
from IPython.display import display, HTML
import copy
output_notebook()

class GLM():
    def __init__(self, data, independent, dependent, weight, family='Poisson'):
        self.data = data
        self.independent = independent
        self.dependent = dependent
        self.weight = weight
        self.base_dict = self.set_base_level()
        self.PDP = None 
        self.variates = {}
        self.customs = {}
        self.transformed_data = self.data[[self.weight] + [self.dependent]]
        self.formula = ''
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
    
    def create_variate(self, name, column, degree):
        sub_dict = {}
        Z, norm2, alpha = self.ortho_poly_fit(x = self.data[column], degree=degree)
        Z = pd.DataFrame(Z)
        Z.columns = [name + '_p' + str(idx) for idx in range(degree + 1)]
        sub_dict['Z'] = Z
        sub_dict['norm2'] = norm2
        sub_dict['alpha'] = alpha
        sub_dict['source'] = column
        self.variates[name] = sub_dict
        
    def create_custom(self, name, column, dictionary):
        temp = self.data[column].map(dictionary)
        temp = pd.get_dummies(temp.to_frame(), drop_first=True)
        self.customs[name] = {'source':column, 'Z':temp}
    
    def fit(self, simple=[], customs=[], variates=[]):
        # Sets PDP dictionary
        # Sets training subset
        simple_str = ' + '.join(simple)
        variate_str = ' + '.join([' + '.join(self.variates[item]['Z'].columns) for item in variates])
        custom_str = ' + '.join([' + '.join(self.customs[item]['Z'].columns) for item in customs])
        if simple_str != '' and variate_str != '':
            variate_str = ' + ' + variate_str 
        if simple_str + variate_str != '' and custom_str != '':
            custom_str = ' + ' + custom_str
        self.formula = self.dependent + ' ~ ' +  simple_str + variate_str + custom_str    
        self.transformed_data = self.data[self.independent + [self.weight] + [self.dependent]] #+ list(set([v['source'] for k,v in self.variates.items() if k in variates] + [v['source'] for k,v in self.customs.items() if k in customs]))
        for i in range(len(variates)):
            self.transformed_data = pd.concat((self.transformed_data, self.variates[variates[i]]['Z']), axis=1)
        for i in range(len(customs)):
            self.transformed_data = pd.concat((self.transformed_data, self.customs[customs[i]]['Z']), axis=1)
        self.model = sm.GLM.from_formula(formula=self.formula , data=self.transformed_data, family=sm.families.Poisson())
        self.results = self.model.fit()
        fitted = self.results.predict(self.transformed_data)
        fitted.name="Fitted Avg"
        self.transformed_data = pd.concat((self.transformed_data, fitted),axis=1)
        self.set_PDP()
        
    
    def ortho_poly_fit(self, x, degree = 1):
        n = degree + 1
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
    
    def ortho_poly_predict(self, x, alpha, norm2, degree = 1):
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
    
    def view(self):
        def view_one_way(var, fitted, model):
            temp = pd.pivot_table(data=self.transformed_data, index=[var], values=['Loss', 'Exposure', 'Fitted Avg'], aggfunc=np.sum)
            temp['Observed'] = temp['Loss']/temp['Exposure']
            temp['Fitted'] = temp['Fitted Avg']/temp['Exposure']
            temp = temp.merge(self.PDP[var][["Model"]], how='inner', left_index=True, right_index=True)
            y_range = Range1d(start=0, end=temp['Exposure'].max()*1.8)
            if type(temp.index) == pd.core.indexes.base.Index:
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
        vw = interactive(view_one_way, var=self.independent, fitted=True, model=True, layout=widgets.Layout(display="inline-flex"))
        return vw

        
        
        



def code_toggle():
    return HTML('''<script>
    code_show=true; 
    function code_toggle() {
     if (code_show){
     $('div.input').hide();
     } else {
     $('div.input').show();
     }
     code_show = !code_show
    } 
    $( document ).ready(code_toggle);
    </script>
    <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')

def independent_set(columns):
    def widget(**kwargs):
        return kwargs
    w = interactive(widget, **{item:False for item in columns})
    display(w)
    return w



