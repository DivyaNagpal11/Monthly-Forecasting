# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:40:07 2020

@author: DN067571
"""
########################## Sarima Model ##########################################
from itertools import product
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly
import plotly.graph_objs as go
import os
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


def optimize_SARIMA(parameters_list,train_set,val_set):
    results=[]
    best_adj_mape = float('inf')
    for param in tqdm_notebook(parameters_list):
        try: 
            model=sm.tsa.statespace.SARIMAX(train_set,order=(param[0],param[1],param[2]),seasonal_order=(param[3],param[4],param[5],param[6])).fit(disp=-1)
            fore1=model.predict(start=train_set.shape[0],end=train_set.shape[0]+len(val_set)-1)
            fore=np.array(fore1)
            
            y_true=np.array(list(train_set[param[6]+param[1]:]['total_charge']))
            y_pred=np.array(list(model.fittedvalues[param[6]+param[1]:]))
            train_mape=round(mean_absolute_percentage_error(y_true,y_pred),2)
            val_mape=round(mean_absolute_percentage_error(val_set['total_charge'],fore),2)
            adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_set))+val_mape*len(val_set)/(len(y_true)+len(val_set))
            if adj_mape <= best_adj_mape:
                best_adj_mape=adj_mape
                best_model = model    
            results.append([param,model.aic,train_mape,val_mape,adj_mape])
        except:
            continue
        
    
    result_table=pd.DataFrame(results)
    result_table.columns=['parameters','aic','train_mape','val_mape','adj_mape']
    result_table=result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
    return result_table, best_model

    
def sarima_model(train_set,val_set,test_set,be_name,set_no):    
#    train_set,val_set,test_set,be_name,set_no=train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name,c
#    train_set,val_set,test_set=new_df_monthly[:-6],new_df_monthly[-6:-3],new_df_monthly[-3:]
    p=range(0,4)
    d=range(0,2)
    q=range(0,4)
    P=range(0,3)
    D=range(0,2)
    Q=range(0,3)
    s=(3,6,12)
    
    parameters=product(p,d,q,P,D,Q,s)
    parameters_list=list(parameters)
    result_table, best_model = optimize_SARIMA(parameters_list,train_set,val_set)

    fore_val=best_model.predict(start=train_set.shape[0],end=train_set.shape[0]+len(val_set)-1)
    
    overall_train=pd.concat([train_set,val_set])
    fitted_val_list=[]
    res_aic_avg = 2* np.abs(result_table.aic.mean())
    
    c=1
    for p1, d1, q1, P1, D1, Q1, s1 in result_table.parameters:
        try:
            if c > 5:
                break
            best_model_overall=sm.tsa.statespace.SARIMAX(overall_train,order=(p1,d1,q1),seasonal_order=(P1,D1,Q1,s1)).fit(disp=-1)
            if best_model_overall.aic < res_aic_avg:
                c=c+1
                fore_test=best_model_overall.predict(start=overall_train.shape[0],end=overall_train.shape[0]+len(test_set)-1)
                fore_test_set=np.array(list(fore_test))
                fitted_val_list.append([fore_test_set[0],fore_test_set[1],fore_test_set[2]])
                
        except:
            continue
    fitted_val=pd.DataFrame(fitted_val_list,columns=['1st','2nd','3rd'])
    fitted_mean=[fitted_val['1st'].mean(),fitted_val['2nd'].mean(),fitted_val['3rd'].mean()]
    
    test_set1=np.array(list(test_set["total_charge"]))
    test_results=round(mean_absolute_percentage_error(test_set1,fitted_mean),2)
    
    one_month=round(mean_absolute_percentage_error(test_set1[0],fitted_mean[0]),2)
    two_month=round(mean_absolute_percentage_error(test_set1[1],fitted_mean[1]),2)
    third_month=round(mean_absolute_percentage_error(test_set1[2],fitted_mean[2]),2)
    
    p, d, q, P, D, Q, s = result_table.parameters[0]
    graph_path_tso = r"graphs_outlier/"+be_name+"/"
    if not os.path.exists(graph_path_tso):
        os.makedirs(graph_path_tso)
    trace1 = go.Scatter(x=train_set[s+d:].index, y=train_set[s+d:]['total_charge'], mode='lines+markers',name="Actual values: Train", marker=dict(color="blue",size=9, line=dict(width=1)))
    trace2 = go.Scatter(x=best_model.fittedvalues[s+d:].index, y=best_model.fittedvalues[s+d:].values, mode='lines+markers',name="Fitted values: Train", marker=dict(color="red",size=9,line=dict(width=1)))
    trace3 = go.Scatter(x=val_set['total_charge'].index, y=val_set['total_charge'], mode='lines+markers',name="Actual values: Val", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace4 = go.Scatter(x=fore_val.index, y=fore_val.values, mode='lines+markers',name="Predicted values: Val", marker=dict(color="rgb(44, 160, 44)",size=9,line=dict(width=1))) 
    trace5 = go.Scatter(x=test_set['total_charge'].index, y=test_set['total_charge'], mode='lines+markers',name="Actual values: Test", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace6 = go.Scatter(x=fore_test.index, y=fitted_mean, mode='lines+markers',name="Predicted values: Test", marker=dict(color="orange",size=9,line=dict(width=1))) 
    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    layout = go.Layout(go.Layout(title='Billing Entity : {} <br>Train MAPE: {} Val MAPE: {} Test MAPE: {} <br>SARIMA(p,d,q)(P,D,Q,s):({},{},{})({},{},{},{})'.format(be_name,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_results,p, d, q, P, D, Q, s),yaxis=dict(title="Monthly Charges", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Month-Year",
                                 ), boxmode='group'))
    fig = go.Figure(data=data,layout=layout)
    plotly.offline.plot(fig, filename=graph_path_tso+str(set_no)+"_sarima", image='png')
    return round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_results,one_month,two_month,third_month,test_set1,fitted_mean


