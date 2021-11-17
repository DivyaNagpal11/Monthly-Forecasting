# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 01:14:44 2020

@author: DN067571
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:43:07 2020

@author: DN067571
"""

from tbats import TBATS
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class TBATS_CLASS:

    def __init__(self,**kwargs):

        self.train,self.test = kwargs.get('train'),kwargs.get('test')       
        self.random_state = kwargs.get('random_state',1)
        self.target_dates = kwargs.get('target_dates')
        self.target_value = kwargs.get('target_value')
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


    def fit(self,data,param):
        estimator = TBATS(
                     seasonal_periods=[3,12],
                     use_arma_errors=param[0],  # shall try models with and without ARMA
                     use_box_cox=param[1],  # will not use Box-Cox
                     use_trend=param[2],  # will try models with trend and without it
                     use_damped_trend=param[3],  # will try models with daming and without it
                     show_warnings=False,  # will not be showing any warnings for chosen model
                     )
             
        fitted_model = estimator.fit(data)
        print(fitted_model.summary())
        return fitted_model


    def forecast(self,model,n_periods):
        return model.forecast(steps=n_periods) 


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
      
        
    def optimize_tbats(self,parameters_list,train_set,val_set):  
        results=[]
        best_adj_mape=float('inf')
        for param in parameters_list[0:5]:
            
            fitted_model = self.fit(train_set[[self.target_value]],param)
#            print("summary",fitted_model.summary())
#            val_predicted=fitted_model.forecast(steps=self.n_periods)
            val_predicted = self.forecast(fitted_model,self.n_periods)
#            print("val_predicted",val_predicted) 
            y_true=np.array(list(train_set[self.target_value]))
#            print("y_true",y_true)
            y_pred=np.array(list(fitted_model.y_hat))
#            print("y_pred",y_pred)
             
            train_mape=round(self.mean_absolute_percentage_error(y_true,y_pred),2)
#            print("train_mape",train_mape)
            val_mape=round(self.mean_absolute_percentage_error(val_set[self.target_value],val_predicted),2)
#            print("val_mape",val_mape)
            adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_set))+val_mape*len(val_set)/(len(y_true)+len(val_set))
#            print("adj_mape",adj_mape)
            if adj_mape <= best_adj_mape:
                best_adj_mape=adj_mape
                best_model = fitted_model    
            results.append([param,fitted_model.aic,train_mape,val_mape,adj_mape])
        
        result_table=pd.DataFrame(results,columns=['parameters','aic','train_mape','val_mape','adj_mape'])
        result_table=result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
        return result_table, best_model
    
             
    def run_model(self):
        print("In Tbats")
        val_set=self.train[-(int(self.n_periods)):]
        result_table, best_model = self.optimize_tbats(self.parameters_list,self.train[:-int(self.n_periods)],val_set)
        overall_train=self.train
        fitted_val_list=[]
        res_aic_avg = 2* np.abs(result_table.aic).mean()
        c=1
        for param in result_table.parameters:
            try:
                if c > 3:
                    break
                
                best_model_overall = self.fit(overall_train,param)
                if best_model_overall.aic < res_aic_avg:
                    c=c+1
                    fore_test_set = self.forecast(best_model_overall,self.n_periods)
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
            
        print("in tbats fitted mean",fitted_mean)
        if self.walk_forward == "True":
            test_set1=np.array(list(self.test[self.target_value]))
            test_mape=round(self.mean_absolute_percentage_error(test_set1,fitted_mean),2)
            one_month=round(self.mean_absolute_percentage_error(test_set1[0],fitted_mean[0]),2)
            two_month=round(self.mean_absolute_percentage_error(test_set1[1],fitted_mean[1]),2)
            third_month=round(self.mean_absolute_percentage_error(test_set1[2],fitted_mean[2]),2)
            self.results.append([self.target_item,"TBATS",self.set_no,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_mape,test_set1,fitted_mean,one_month,two_month,third_month])
            self.mape_res=pd.DataFrame(self.results,columns=[self.target_items,'model','set_no','train_mape','val_mape','test_mape','test_actual','test_predicted','one_month','two_month','third_month'])
        else:
            self.results.append([self.target_item,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),list(self.test[self.target_dates]),fitted_mean])
            self.mape_res=pd.DataFrame(self.results,columns=[self.target_items,'train_mape','val_mape','test_dates','test_predicted'])
    
        
        
   