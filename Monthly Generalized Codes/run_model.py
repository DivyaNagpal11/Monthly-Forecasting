# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:53:01 2020

@author: DN067571
"""

import numpy as np
from model.models import Univariate
import json
import argparse
import pickle
import pandas as pd
from os import path,makedirs
from sklearn import preprocessing
save_path ='model_output/'


########################### For walk forward ##################################
def filter_data(ti):
    subset_df = preprocessed_data[preprocessed_data[target_items]==ti].reset_index()
#    print("subset_df",subset_df.head(5))
    subset_df[target_dates] = pd.to_datetime(subset_df[target_dates])
    subset_df = subset_df.sort_values(target_dates)#.reset_index()
    subset_df=subset_df.drop([target_items],axis=1)
    subset_df.set_index(target_dates,inplace=True)
#    print("subset_df",subset_df.head(5))
    return subset_df


def generate_best_model(res_df):
    be_names=list(res_df[target_items].unique())
    model_names=list(res_df.model.unique())
    best_model_list=[]
    
    for m in be_names:
        subset_df=res_df[res_df[target_items] == m]
        models_avg=[]
        for n in model_names:
            model_df=subset_df[subset_df.model == n]
            test_mape_avg = model_df.test_mape.mean()
            models_avg.append(test_mape_avg)
        min_mape=min(models_avg)
        min_mape_index=models_avg.index(min_mape)
        best_model=model_names[min_mape_index]
        best_model_list.append([m,best_model,round(min_mape,2)])
        
    best_model_list=pd.DataFrame(best_model_list,columns=[target_items,"best_model","mape_avg_min"])
    return best_model_list


def walk_forward_func(wf_results):
    out_path=save_path+"wf_results/"
    for ti in tis:
        set_no=wf_results[wf_results[target_items]==ti]["set_no"].max()+1
        subset_df=filter_data(ti)
        print("model_name",args['model_name'])
        models = Univariate(ti,model_param,data = subset_df,model_name=args['model_name'],target_items=target_items,target_value=target_value,other_target_value=other_target_value,target_dates=target_dates,n_periods=data_config['n_periods'],set_no=set_no,wf=args['walk_forward'])
        wf_results=wf_results.append(models.models_ran)
    
    
    if path.exists(out_path+args['target_item'])==False:
        makedirs(out_path+args['target_item'])
        
    obj_file = out_path+args['target_item']+'/'+data_date+'.csv'
    
    best_model_list=generate_best_model(wf_results) 
    best_model_list.to_csv(obj_file)
    
    with open('data-config.json') as f1:
        config_file = json.load(f1)
    f1.close()
    config_file['best_model_path'] = obj_file
    with open('data-config.json',"w") as f2:
        f2.write(json.dumps(config_file))
    f2.close()


########################### For Production Code ###############################
def report_generation1(report_data,best_model_res,timestampStr,ti,n_periods):
    x1=report_data[[target_dates,target_value]][:-int(n_periods)]
    x1["prediction"]='FALSE'
    x1[target_dates]=[i.strftime("%d-%b-%Y") for i in x1[target_dates]]
    x2=pd.DataFrame(best_model_res['test_predicted'][0][:],timestampStr)
    x2["prediction"]='TRUE'
    x3 = x2.reset_index()
    x3=x3.rename(columns = {"index": target_dates, 0:target_value})
    x4=x1.append(x3).reset_index(drop=True)
    x5=x4.replace(np.nan, '', regex=True)
    x5[target_items]=ti
    print("x5",x5)
    return x5


def production_code():
    out_path=save_path+"prod/"
    final_report=pd.DataFrame()
    with open('model-config.json') as f:
        model_config = json.load(f)
    best_model_path=model_config['best_model_path']
    best_model_file=pd.read_csv(best_model_path)
    for ti in tis:
        best_model_name=str.lower(best_model_file[best_model_file[target_items]==ti]["best_model"].values[0])
        subset_df = preprocessed_data[preprocessed_data[target_items]==ti].reset_index()
        subset_df=subset_df.drop([target_items],axis=1)
        models = Univariate(ti,model_param,data = subset_df,target_items=target_items,target_value=target_value,other_target_value=other_target_value,target_dates=target_dates,n_periods=model_config['n_periods'],model_name=best_model_name,wf=args['walk_forward'])
        timestampStr =[i.strftime("%d-%b-%Y") for i in models.models_ran["test_dates"][0][:]]
        report=report_generation1(subset_df,models.models_ran,timestampStr,ti,model_config['n_periods'])
        final_report=final_report.append(report)
        
    if path.exists(out_path+args['target_item'])==False:
        makedirs(out_path+args['target_item'])   
    obj_file = out_path+args['target_item']+'/'+data_date+'.csv'
    final_report.to_csv(obj_file)
        

############################# Running the scripts ############################    
parser = argparse.ArgumentParser(description='Run Models for processed files')
parser.add_argument("-ti", "--target_item", 
                    required=True, 
                    help="Target_Item eg:- mention target item you want to run for or default give value as 'All' ")
parser.add_argument("-wf", "--walk_forward",required=True)
parser.add_argument("-mn", "--model_name",required=False, default='All')

with open('data-config.json') as f:
  data_config = json.load(f)

with open('model_params.json') as x:
  model_params = json.load(x)
  
args = vars(parser.parse_args())
model_param=model_params['model_params']

filehandler = open(data_config['processed_data_path'], 'rb') 
data_object = pickle.load(filehandler)
data_date = data_object.summary_df.end_date.max().month_name() +'-'+str(data_object.summary_df.end_date.max().year)
preprocessed_data  = data_object.transform_data

target_dates = data_object.target_dates
target_value = data_object.target_value
target_items = data_object.target_items
other_target_value = data_object.other_target_value

if args['target_item'] == 'All':
    tis = preprocessed_data[target_items].unique()

else:
    tis = [args['target_item']]

if args['walk_forward'] == "True":
    wf_res_file=data_config["wf_res_path"]
    wf_results=pd.read_csv(wf_res_file)
    walk_forward_func(wf_results)
else:
    production_code()
    
    


 
