# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:03:06 2020

@author: DN067571
"""

from itertools import product
from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import os
from fbprophet import Prophet
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
  
def optimize_prophet(parameters_list,train_dataset,val_dataset,steps):  
    results=[]
    best_adj_mape=float('inf')
    for i in tqdm(parameters_list):
#        try:
        forecast=pd.DataFrame()
        future=pd.DataFrame()
        
        prophet_basic = Prophet(growth='linear',daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=False,n_changepoints=20,changepoint_prior_scale=i[0])
        prophet_basic.add_seasonality(name='quarterly', period=3, fourier_order=i[1]).add_seasonality(name='biannual', period=6, fourier_order=i[2]).add_seasonality(name='yearly', period=12, fourier_order=i[3])
        prophet_basic.fit(train_dataset)
        
        future= prophet_basic.make_future_dataframe(periods=3)
        forecast=prophet_basic.predict(future)
        
        y_true=np.array(list(train_dataset['y']))
        y_pred=np.array(list(forecast.yhat[:-steps]))
        val_predicted=np.array(list(forecast.yhat[-steps:]))
        train_mape=round((mean_absolute_percentage_error(y_true,y_pred)),2)
        val_mape=round((mean_absolute_percentage_error(val_dataset["y"],val_predicted)),2)
        adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_dataset))+val_mape*len(val_dataset)/(len(y_true)+len(val_dataset))
        
        if adj_mape <= best_adj_mape:
            best_adj_mape=adj_mape
            best_model = prophet_basic
            
        results.append([i,train_mape,val_mape,adj_mape])
#        except:
#            continue
        
    result_table=pd.DataFrame(results,columns=['parameters','train_mape','val_mape','adj_mape'])
    result_table=result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
    return result_table, best_model


def monthly_prophet(train_set,val_set,test_set,be_name,set_no):
#    train_set,val_set,test_set,be_name,set_no=train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name,c
    train_dataset= pd.DataFrame()
    val_dataset= pd.DataFrame()
    train_set=train_set.reset_index()
    val_set=val_set.reset_index()
    train_dataset['ds'] = train_set["posted_date"]
    train_dataset['y']=train_set["total_charge"]
    val_dataset['ds'] = val_set["posted_date"]
    val_dataset['y']=val_set["total_charge"]
    steps = len(val_set)
      
    cp=(.3,.4,.5)
    fo3=(4,8,10)
    fo6=(4,8)
    fo12=(4,6,8,10)
    
    parameters=product(cp,fo3,fo6,fo12)
    parameters_list=list(parameters)
    result_table, best_model = optimize_prophet(parameters_list,train_dataset,val_dataset,steps)
    future= best_model.make_future_dataframe(periods=3)
    forcast_val=best_model.predict(future).yhat[-3:]
    
    overall_train=pd.concat([train_set,val_set])
    overall_train['ds'] = overall_train.posted_date
    overall_train['y'] = overall_train["total_charge"]
    fitted_val_list=[]
#    res_aic_avg = 2* np.abs(result_table.aic.mean())
    
    c=1
    for cp,fo3,fo6,fo12 in result_table.parameters:
        try:
            if c > 5:
                break
            prophet_basic1 = Prophet(growth='linear',daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=False,n_changepoints=20,changepoint_prior_scale=cp)
            prophet_basic1.add_seasonality(name='quarterly', period=3, fourier_order=fo3).add_seasonality(name='biannual', period=6, fourier_order=fo6).add_seasonality(name='yearly', period=12, fourier_order=fo12)
            prophet_basic1.fit(overall_train)
#            if prophet_basic1.aic < res_aic_avg:
            future= prophet_basic1.make_future_dataframe(periods=3)
            forecast=prophet_basic1.predict(future).yhat[-3:]
            c=c+1
            fitted_val_list.append([forecast.iloc[0],forecast.iloc[1],forecast.iloc[2]])
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
    trace1 = go.Scatter(x=train_set["posted_date"], y=train_set['total_charge'], mode='lines+markers',name="Actual values: Train", marker=dict(color="blue",size=9, line=dict(width=1)))
    trace2 = go.Scatter(x=train_set["posted_date"], y=best_model.predict(future).yhat[:-steps], mode='lines+markers',name="Fitted values: Train", marker=dict(color="red",size=9,line=dict(width=1)))
    trace3 = go.Scatter(x=val_set['posted_date'], y=val_set['total_charge'], mode='lines+markers',name="Actual values: Val", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace4 = go.Scatter(x=val_set['posted_date'], y=forcast_val, mode='lines+markers',name="Predicted values: Val", marker=dict(color="rgb(44, 160, 44)",size=9,line=dict(width=1))) 
    trace5 = go.Scatter(x=test_set.index, y=test_set['total_charge'], mode='lines+markers',name="Actual values: Test", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace6 = go.Scatter(x=test_set.index, y=fitted_mean, mode='lines+markers',name="Predicted values: Test", marker=dict(color="orange",size=9,line=dict(width=1))) 
    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    layout = go.Layout(go.Layout(title='Billing Entity : {} <br>Train MAPE: {} Val MAPE: {} Test MAPE: {} <br>PROPHET'.format(be_name,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_mape),yaxis=dict(title="Monthly Charges", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Month-Year",
                                 ), boxmode='group'))
    fig = go.Figure(data=data,layout=layout)
    plotly.offline.plot(fig, filename=graph_path_tso+str(set_no)+"_prophet", image='png')
    
    return round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_mape,one_month,two_month,third_month,test_set1,fitted_mean
    
