# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 01:52:47 2020

@author: DN067571
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:05:45 2020

@author: DN067571
"""

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class VARMAX_MODEL:

    def __init__(self,**kwargs):

        self.train,self.test = kwargs.get('train'),kwargs.get('test')       
        self.random_state = kwargs.get('random_state',1)
        self.target_dates = kwargs.get('target_dates')
        self.target_value = kwargs.get('target_value')
        self.other_target_value = kwargs.get('other_target_value')
        self.target_item = kwargs.get('ti')
        self.target_items = kwargs.get('target_items')
        self.set_no = kwargs.get('set_no')
        self.parameters_list = kwargs.get('parameters_list',None)
        self.walk_forward = kwargs.get('wf')
        self.n_periods = int(kwargs.get('n_periods'))

        #output values
        self.fitted_values = pd.DataFrame()
        self.test_prediction = pd.DataFrame()
        self.unseen_prediction = pd.DataFrame()        
        self.apes = []
        self.results=[]
        self.mape_res=pd.DataFrame()
        self.run_model()
        del self.train,self.test


    def fit(self,data,exog_data,exog_cols,param):
        return VARMAX(data[[self.target_value,self.other_target_value]], exog=exog_data.reset_index()[exog_cols], order=(param[0],param[1])).fit(disp=-1)


    def forecast(self,model,exog_data,exog_cols,n_periods):
        return model.forecast(steps=n_periods,exog=exog_data.reset_index()[exog_cols]) 


    def split_train_val(self,subset_df):
        return subset_df[:(-self.n_periods)],subset_df[(-self.n_periods):]
 

    def calculate_apes(self):
        for i,j in zip(self.test[self.target_value].values.flatten(),self.test_prediction.values.flatten()):            
            self.apes.append(self.mean_absolute_percentage_error(i,j))
            

    def mean_absolute_percentage_error(self,y_true,y_pred):
        y_true=pd.Series(y_true)
        y_pred=pd.Series(y_pred)
        true_pred = pd.DataFrame(zip(y_true,y_pred),columns=['y_true','y_pred'])
        true_pred.drop(true_pred[true_pred['y_pred'] == 0].index, axis=0, inplace=True)
        true_pred.drop(true_pred[true_pred['y_true'] == 0].index, axis=0, inplace=True)
        return np.mean(np.abs(np.subtract(true_pred.y_true,true_pred.y_pred)/true_pred.y_true))*100
    
    def optimize_VARMAX(self,parameters_list,train_set_transformed,train_set,val_set,scaler,exog_cols):   
        results=[]
        best_adj_mape = float('inf')
        for param in parameters_list:
            try:
                
                model=self.fit(train_set_transformed,train_set,exog_cols,param)
                fore1=self.forecast(model,val_set,exog_cols,self.n_periods)
                fore=scaler.inverse_transform(fore1)[:,0]
                
                y_true=scaler.inverse_transform(train_set_transformed)[:,0]
                y_pred=scaler.inverse_transform(model.fittedvalues)[:,0]
                
                train_mape=round(self.mean_absolute_percentage_error(y_true,y_pred),2)
                val_mape=round(self.mean_absolute_percentage_error(val_set[self.target_value],fore),2)
                adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_set))+val_mape*len(val_set)/(len(y_true)+len(val_set))
                if adj_mape <= best_adj_mape:
                    best_adj_mape=adj_mape
                    best_model = model    
                results.append([param,model.aic,train_mape,val_mape,adj_mape])
            except:
                continue
        
        result_table=pd.DataFrame(results,columns=['parameters','aic','train_mape','val_mape','adj_mape'])
        result_table=result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
        return result_table,best_model  
    
    
#    def optimize_VARMAX(self,parameters_list,train_set_transformed,train_set,val_set,scaler,exog_cols):   
#        results=[]
#        best_adj_mape = float('inf')
#        for param in tqdm_notebook(parameters_list):
#            try:
#                model=self.fit(train_set_transformed,train_set[exog_cols],param)
#                fore1=model.forecast(steps=self.n_periods,exog=val_set[exog_cols])
#                fore=scaler.inverse_transform(fore1)[:,0]
#                
#                y_true=scaler.inverse_transform(train_set_transformed)[:,0]
#                y_pred=scaler.inverse_transform(model.fittedvalues)[:,0]
#                
#                train_mape=round(self.mean_absolute_percentage_error(y_true,y_pred),2)
#                val_mape=round(self.mean_absolute_percentage_error(val_set[self.target_value],fore),2)
#                adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_set))+val_mape*len(val_set)/(len(y_true)+len(val_set))
#                if adj_mape <= best_adj_mape:
#                    best_adj_mape=adj_mape
#                    best_model = model    
#                results.append([param,model.aic,train_mape,val_mape,adj_mape])
#            except:
#                continue
#        
#        result_table=pd.DataFrame(results,columns=['parameters','aic','train_mape','val_mape','adj_mape'])
#        result_table=result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
#        return result_table,best_model
#    
    def run_model(self):
        print("In varmax")
        scaler = MinMaxScaler()
        varmax_columns = [self.target_value,self.other_target_value]
        train_set=self.train[:-self.n_periods]
        val_set=self.train[-self.n_periods:]
        
        scaler.fit(train_set[varmax_columns])
        train_set_transformed = pd.DataFrame(scaler.transform(train_set[varmax_columns]))
        train_set_transformed.columns=varmax_columns
#        print("transformed",train_set_transformed.head(5))
        
        exog_columns=self.train.columns
        exog_columns=exog_columns.drop([self.target_value,self.other_target_value])
        result_table, best_model = self.optimize_VARMAX(self.parameters_list,train_set_transformed,train_set,val_set,scaler,exog_columns)
        
#        fore_val= best_model.forecast(steps=self.n_periods,exog=val_set[exog_columns])
#        fore_val=scaler.inverse_transform(fore_val)[:,0]
#        fitted_values=scaler.inverse_transform(best_model.fittedvalues)[:,0]
        
        overall_train=self.train
        scaler.fit(overall_train[varmax_columns])
        overall_train_transformed = pd.DataFrame(scaler.transform(overall_train[varmax_columns]))
        overall_train_transformed.columns=varmax_columns
        
        fitted_val_list=[]
        res_aic_avg = 2* np.abs(result_table.aic.mean())
        c=1
        for param in result_table.parameters:
            try:
                if c > 3:
                    break
                best_model_overall=self.fit(overall_train_transformed,overall_train,exog_columns,param)
                if best_model_overall.aic < res_aic_avg:
                    c=c+1
                    fore_test=self.forecast(best_model_overall,self.test,exog_columns,self.n_periods)
                    fore_test_set=scaler.inverse_transform(fore_test)[:,0]
                    
                    get_list=[]
                    for i in range(len(fore_test_set)):
                        get_list.append(fore_test_set[i])
                    fitted_val_list.append(get_list)   
            except:
                continue
       
        fitted_val=pd.DataFrame(fitted_val_list,columns=np.arange(1,self.n_periods+1))
        fitted_mean=[]
        for i in np.arange(1,self.n_periods+1):
            fitted_mean.append(fitted_val[i].mean())
            
#        print("in varmax fitted mean",fitted_mean)
        if self.walk_forward == "True":
            test_set1=np.array(list(self.test[self.target_value]))
            test_mape=round(self.mean_absolute_percentage_error(test_set1,fitted_mean),2)
            one_month=round(self.mean_absolute_percentage_error(test_set1[0],fitted_mean[0]),2)
            two_month=round(self.mean_absolute_percentage_error(test_set1[1],fitted_mean[1]),2)
            third_month=round(self.mean_absolute_percentage_error(test_set1[2],fitted_mean[2]),2)
            self.results.append([self.target_item,"VARMAX",self.set_no,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_mape,test_set1,fitted_mean,one_month,two_month,third_month])
            self.mape_res=pd.DataFrame(self.results,columns=[self.target_items,'model','set_no','train_mape','val_mape','test_mape','test_actual','test_predicted','one_month','two_month','third_month'])
        else:
            self.results.append([self.target_item,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),list(self.test[self.target_dates]),fitted_mean])
            self.mape_res=pd.DataFrame(self.results,columns=[self.target_items,'train_mape','val_mape','test_dates','test_predicted'])
    