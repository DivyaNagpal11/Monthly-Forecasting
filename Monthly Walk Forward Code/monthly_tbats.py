# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:43:07 2020

@author: DN067571
"""

from tbats import TBATS
from itertools import product
import numpy as np
import os
import pandas as pd
import plotly
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

def mean_absolute_percentage_error(y_true,y_pred):
    y_true=pd.Series(y_true)
    y_pred=pd.Series(y_pred)
    true_pred = pd.DataFrame(zip(y_true,y_pred),columns=['y_true','y_pred'])
    true_pred.drop(true_pred[true_pred['y_pred'] == 0].index, axis=0, inplace=True)
    true_pred.drop(true_pred[true_pred['y_true'] == 0].index, axis=0, inplace=True)
    return np.mean(np.abs(np.subtract(true_pred.y_true,true_pred.y_pred)/true_pred.y_true))*100
#      return np.mean(np.abs(np.subtract(y_true,y_pred)/y_true))*100
  
    
def optimize_tbats(parameters_list,train_set,val_set,steps):  
    results=[]
    best_adj_mape=float('inf')
    for i in parameters_list:
         estimator = TBATS(
                 seasonal_periods=[3,12],
                 use_arma_errors=i[0],  # shall try models with and without ARMA
                 use_box_cox=i[1],  # will not use Box-Cox
                 use_trend=i[2],  # will try models with trend and without it
                 use_damped_trend=i[3],  # will try models with daming and without it
                 show_warnings=False,  # will not be showing any warnings for chosen model
                 )
         
         fitted_model = estimator.fit(train_set)
         val_predicted = fitted_model.forecast(steps=steps)
        
         y_true=np.array(list(train_set['total_charge']))
         y_pred=np.array(list(fitted_model.y_hat))
         
         train_mape=round(mean_absolute_percentage_error(y_true,y_pred),2)
         val_mape=round(mean_absolute_percentage_error(val_set["total_charge"],val_predicted),2)
         adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_set))+val_mape*len(val_set)/(len(y_true)+len(val_set))
         if adj_mape <= best_adj_mape:
             best_adj_mape=adj_mape
             best_model = fitted_model    
         results.append([i,fitted_model.aic,train_mape,val_mape,adj_mape])
    
    result_table=pd.DataFrame(results,columns=['parameters','aic','train_mape','val_mape','adj_mape'])
    result_table=result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
    return result_table, best_model

         
def monthly_tbats(train_set,val_set,test_set,be_name,set_no):
#    train_set,val_set,test_set,be_name=train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name
    steps = 3
    np.random.seed(2342)
      
    use_arma_errors=[True,False]
    use_box_cox=[True,False]
    use_trend=[True,False]
    use_damped_trend=[True,False]  
    
    parameters=product(use_arma_errors,use_box_cox,use_trend,use_damped_trend)
    parameters_list=list(parameters)
    result_table, best_model = optimize_tbats(parameters_list,train_set,val_set,steps)

    forcast_val=best_model.forecast(steps=steps)
    
    overall_train=pd.concat([train_set,val_set])
    fitted_val_list=[]
    res_aic_avg = 2* np.abs(result_table.aic).mean()
    c=1
    for i in result_table.parameters:
        try:
            if c > 3:
                break
            estimator = TBATS(
                    seasonal_periods=[3,12],
                    use_arma_errors=i[0],  # shall try models with and without ARMA
                    use_box_cox=i[1],  # will not use Box-Cox
                    use_trend=i[2],  # will try models with trend and without it
                    use_damped_trend=i[3],  # will try models with daming and without it
                    show_warnings=False,  # will not be showing any warnings for chosen model
                    )
            
            best_model_overall = estimator.fit(overall_train)
            print(best_model_overall.aic)
            if best_model_overall.aic < res_aic_avg:
                
                c=c+1
                test_predicted = best_model_overall.forecast(steps=steps)
                fitted_val_list.append([test_predicted[0],test_predicted[1],test_predicted[2]])
            
        except:
            continue
    fitted_val=pd.DataFrame(fitted_val_list,columns=['1st','2nd','3rd'])
    fitted_mean=[fitted_val['1st'].mean(),fitted_val['2nd'].mean(),fitted_val['3rd'].mean()]
    
    test_set1=np.array(list(test_set["total_charge"]))
    test_mape=round(mean_absolute_percentage_error(test_set1,fitted_mean),2)
    
    one_month=round(mean_absolute_percentage_error(test_set1[0],fitted_mean[0]),2)
    two_month=round(mean_absolute_percentage_error(test_set1[1],fitted_mean[1]),2)
    third_month=round(mean_absolute_percentage_error(test_set1[2],fitted_mean[2]),2)
    
    graph_path_tso = r"graphs_outlier/"+be_name+"/"
    if not os.path.exists(graph_path_tso):
        os.makedirs(graph_path_tso)
    trace1 = go.Scatter(x=train_set.index, y=train_set['total_charge'], mode='lines+markers',name="Actual values: Train", marker=dict(color="blue",size=9, line=dict(width=1)))
    trace2 = go.Scatter(x=train_set.index, y=best_model.y_hat, mode='lines+markers',name="Fitted values: Train", marker=dict(color="red",size=9,line=dict(width=1)))
    trace3 = go.Scatter(x=val_set['total_charge'].index, y=val_set['total_charge'], mode='lines+markers',name="Actual values: Val", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace4 = go.Scatter(x=val_set.index, y=forcast_val, mode='lines+markers',name="Predicted values: Val", marker=dict(color="rgb(44, 160, 44)",size=9,line=dict(width=1))) 
    trace5 = go.Scatter(x=test_set.index, y=test_set['total_charge'], mode='lines+markers',name="Actual values: Test", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace6 = go.Scatter(x=test_set.index, y=fitted_mean, mode='lines+markers',name="Predicted values: Test", marker=dict(color="orange",size=9,line=dict(width=1))) 
    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    layout = go.Layout(go.Layout(title='Billing Entity : {} <br>Train MAPE: {} Val MAPE: {} Test MAPE: {} <br>TBATS'.format(be_name,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_mape),yaxis=dict(title="Monthly Charges", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Month-Year",
                                 ), boxmode='group'))
    fig = go.Figure(data=data,layout=layout)
    plotly.offline.plot(fig, filename=graph_path_tso+str(set_no)+"_tbats", image='png')
    
    return round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_mape,one_month,two_month,third_month,test_set1,fitted_mean
    
    
        
   