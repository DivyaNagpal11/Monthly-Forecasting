# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:34:11 2020

@author: DN067571
"""

# -*- coding: utf-8 -*-


import pandas as pd
import time
from itertools import product
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

from model.SARIMAX import SARIMAX
from model.SARIMA import SARIMA
from model.TBATS_MODEL import TBATS_CLASS
from model.HOLTWINTERS import HOLTWINTERS
from model.VARMA import VARMA
from model.VARMAX import VARMAX_MODEL
from model.PROPHET import PROPHET

class Univariate():
    
    def split_train_test(self,subset_df,n_peroids):
        return subset_df[:-n_peroids],subset_df[-n_peroids:]
    
    
    def run_model(self,model_name,**kwargs):  
        
        if model_name == 'sarima':
            try:
                starttime = time.time()
                list=self.model_param[model_name].values()
                parameters_list = [a for a in product(*list)]
                df = SARIMA(train = self.train,test = self.test,parameters_list=parameters_list,ti=self.target_item,**kwargs)
                print("in sarima model.py results",df.mape_res)
                self.models_ran=pd.concat([self.models_ran,df.mape_res])
                print("In model.py time for model ran",time.time()-starttime)
                self.models_timetaken['sarima'] = time.time()-starttime
                return time.time()-starttime
            except:
                print('Dint run sarima model for:',self.target_item)
                return 0
                
        elif model_name == 'sarimax':
            try:
                starttime = time.time()
                list=self.model_param[model_name].values()
                parameters_list = [a for a in product(*list)]
                df = SARIMAX(train = self.train,test = self.test,parameters_list=parameters_list,ti=self.target_item,**kwargs)
                print("in sarimax model.py results",df.mape_res)
                self.models_ran=pd.concat([self.models_ran,df.mape_res])
                print("In model.py time for sarimax model ran",time.time()-starttime)
                self.models_timetaken['sarimax'] = time.time()-starttime
                return time.time()-starttime
            except:
                print('Dint run sarimax model for:',self.target_item)
                return 0
        
        elif model_name == 'holtwinters':
            try:
                starttime = time.time()
                list=self.model_param[model_name].values()
                parameters_list = [a for a in product(*list)]
                df = HOLTWINTERS(train = self.train,test = self.test,parameters_list=parameters_list,ti=self.target_item,**kwargs)
                print("in holtwinter model.py results",df.mape_res)
                self.models_ran=pd.concat([self.models_ran,df.mape_res])
                print("In model.py time for holtwinter model ran",time.time()-starttime)
                self.models_timetaken['holtwinters'] = time.time()-starttime
                return time.time()-starttime
            except:
                print('Dint run holtwinters model for:',self.target_item)
                return 0
        
        elif model_name == 'tbats':
            starttime = time.time()
            param_list=self.model_param[model_name].values()
            parameters_list = [a for a in product(*param_list)]
            print("params_list",parameters_list)
            kwargs['parameters_list'] = parameters_list
            kwargs['train'] = self.train
            kwargs['test'] = self.test
            kwargs['ti'] = self.target_item
            print("kwargs",kwargs.keys())
#            train = self.train,test = self.test,parameters_list=parameters_list,ti=self.target_item,
            df = TBATS_CLASS(**kwargs)
            
            print("in tbats model.py results",df.mape_res)
            self.models_ran=pd.concat([self.models_ran,df.mape_res])
            print("In model.py time for tbats model ran",time.time()-starttime)
            self.models_timetaken['tbats'] = time.time()-starttime
            return time.time()-starttime
                
#            try:
#                
#            except:
#                print('Dint run tbats model for:',self.target_item)
#                return 0
         
        elif model_name == 'varma':
            try:
                starttime = time.time()
                list=self.model_param[model_name].values()
                parameters_list = [a for a in product(*list)]
                df = VARMA(train = self.train,test = self.test,parameters_list=parameters_list,ti=self.target_item,**kwargs)
                print("in varma model.py results",df.mape_res)
                self.models_ran=pd.concat([self.models_ran,df.mape_res])
                print("In model.py time for varma model ran",time.time()-starttime)
                self.models_timetaken['varma'] = time.time()-starttime
                return time.time()-starttime
            except:
                print('Dint run varma model for:',self.target_item)
                return 0
                
        elif model_name == 'varmax':
            try:
                starttime = time.time()
                list=self.model_param[model_name].values()
                parameters_list = [a for a in product(*list)]
                df = VARMAX_MODEL(train = self.train,test = self.test,parameters_list=parameters_list,ti=self.target_item,**kwargs)
                print("in varmax model.py results",df.mape_res)
                self.models_ran=pd.concat([self.models_ran,df.mape_res])
                print("In model.py time for varmax model ran",time.time()-starttime)
                self.models_timetaken['varmax'] = time.time()-starttime
                return time.time()-starttime
            except:
                print('Dint run varmax model for:',self.target_item)
                return 0
                
        elif model_name == 'prophet':
            try:
                starttime = time.time()
                list=self.model_param[model_name].values()
                parameters_list = [a for a in product(*list)]
                df = PROPHET(train = self.train,test = self.test,parameters_list=parameters_list,ti=self.target_item,**kwargs)
                print("in prophet model.py results",df.mape_res)
                self.models_ran=pd.concat([self.models_ran,df.mape_res])
                print("In model.py time for prophet model ran",time.time()-starttime)
                self.models_timetaken['prophet'] = time.time()-starttime
                return time.time()-starttime
            except:
                print('Dint run prophet model for:',self.target_item)
                return 0
            
    def __init__(self,target_item,model_params,**kwargs):
        self.random_state = kwargs.get('random_state',1)
        self.target_dates = kwargs.get('target_dates')
        self.target_value = kwargs.get('target_value')
        self.other_target_value = kwargs.get('other_target_value')
        self.n_periods = int(kwargs.get('n_periods'))
        self.model_name = kwargs.get('model_name','All')
        self.set_no = kwargs.get('set_no',0)
        self.walk_forward=kwargs.get('wf',"True")
        self.train,self.test = self.split_train_test(kwargs.get('data'),self.n_periods)
        self.target_item = target_item
        self.target_items = kwargs.get("target_items")
        print("ti & tis",self.target_item,self.target_items)
        self.model_param = model_params
        self.models_ran = pd.DataFrame()
        self.models_timetaken = {}

        if self.model_name == 'All':
            model_names=list(self.model_param.keys())
            print("model_names",model_names)
            for i in model_names:
                print(i)
#                self.model_name=i
                var_res=self.run_model(i,**kwargs)
                print("For model:",i)
                print("var_res",var_res)
#                time.sleep(50)
#                print("Sleep over")
        else:
            self.run_model(**kwargs)
                
               
        
        
             
           