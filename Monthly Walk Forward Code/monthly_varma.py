# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:31:24 2020

@author: DN067571
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:05:45 2020

@author: DN067571
"""

from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.preprocessing import MinMaxScaler
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

      
def optimize_VARMA(parameters_list,train_set_transformed,train_set,val_set,scaler):   
    results=[]
    best_adj_mape = float('inf')
    for param in tqdm_notebook(parameters_list):
        try:
            model=VARMAX(train_set_transformed[["total_charge","footfall"]], order=(param[0],param[1])).fit(disp=-1)
            fore1=model.forecast(steps=len(val_set))
#            fore=np.array(fore1["total_charge"])
            fore=scaler.inverse_transform(fore1)[:,0]
            
#            y_true=np.array(list(train_set['total_charge']))
#            y_pred=np.array(list(model.fittedvalues['total_charge']))
            
            y_true=scaler.inverse_transform(train_set_transformed)[:,0]
            y_pred=scaler.inverse_transform(model.fittedvalues)[:,0]
            
            
            train_mape=round(mean_absolute_percentage_error(y_true,y_pred),2)
            val_mape=round(mean_absolute_percentage_error(val_set["total_charge"],fore),2)
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

def varma_model(train_set,val_set,test_set,be_name,set_no):
#    train_set,val_set,test_set,be_name,set_no=train,val,test,be_name,c
    scaler = MinMaxScaler()
    varmax_columns = ['total_charge', 'footfall']
    
    scaler.fit(train_set[varmax_columns])
    train_set_transformed = pd.DataFrame(scaler.transform(train_set[varmax_columns]))
    train_set_transformed.columns=varmax_columns
    p=range(1,5)
    q=range(0,5)
    
    parameters=product(p,q)
    parameters_list=list(parameters)
    
    result_table, best_model = optimize_VARMA(parameters_list,train_set_transformed,train_set,val_set,scaler)
    
    fore_val= best_model.forecast(steps=len(val_set))
    fore_val=scaler.inverse_transform(fore_val)[:,0]
    fitted_values=scaler.inverse_transform(best_model.fittedvalues)[:,0]
    
    overall_train=pd.concat([train_set,val_set])
    scaler.fit(overall_train[varmax_columns])
    overall_train_transformed = pd.DataFrame(scaler.transform(overall_train[varmax_columns]))
    overall_train_transformed.columns=varmax_columns
    
    fitted_val_list=[]
    res_aic_avg = 2* np.abs(result_table.aic.mean())
    c=1
    for p1, q1 in result_table.parameters:
        try:
            if c > 3:
                break
            best_model_overall=VARMAX(overall_train_transformed[["total_charge","footfall"]], order=(p1,q1)).fit(disp=-1)
            if best_model_overall.aic < res_aic_avg:
                c=c+1
                fore_test=best_model_overall.forecast(steps=len(test_set))
    #            fore_test_set=np.array(fore_test["total_charge"])
                fore_test_set=scaler.inverse_transform(fore_test)[:,0]
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
    
    p,q = result_table.parameters[0]
    graph_path_tso = r"graphs_outlier/"+be_name+"/"
    if not os.path.exists(graph_path_tso):
        os.makedirs(graph_path_tso)
    trace1 = go.Scatter(x=train_set.index, y=train_set['total_charge'], mode='lines+markers',name="Actual values: Train", marker=dict(color="blue",size=9, line=dict(width=1)))
    trace2 = go.Scatter(x=train_set.index, y=fitted_values, mode='lines+markers',name="Fitted values: Train", marker=dict(color="red",size=9,line=dict(width=1)))
    trace3 = go.Scatter(x=val_set['total_charge'].index, y=val_set['total_charge'], mode='lines+markers',name="Actual values: Val", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace4 = go.Scatter(x=val_set['total_charge'].index, y=fore_val, mode='lines+markers',name="Predicted values: Val", marker=dict(color="rgb(44, 160, 44)",size=9,line=dict(width=1))) 
    trace5 = go.Scatter(x=test_set.index, y=test_set['total_charge'], mode='lines+markers',name="Actual values: Test", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace6 = go.Scatter(x=test_set.index, y=fitted_mean, mode='lines+markers',name="Predicted values: Test", marker=dict(color="orange",size=9,line=dict(width=1))) 
    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    layout = go.Layout(go.Layout(title='Billing Entity : {} <br>Train MAPE: {} Val MAPE: {} Test MAPE: {} <br>VARMA(p,q):({},{})'.format(be_name,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_results,p,q),yaxis=dict(title="Monthly Charges", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Month-Year",
                                 ), boxmode='group'))
    fig = go.Figure(data=data,layout=layout)
    plotly.offline.plot(fig, filename=graph_path_tso+str(set_no)+"_varma", image='png')
    return round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_results,one_month,two_month,third_month,test_set1,fitted_mean