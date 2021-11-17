# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:47:30 2020

@author: DN067571
"""

import pandas as pd
import numpy as np
from datetime import datetime
import collections
import warnings
warnings.filterwarnings('ignore')
import holidays
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import statistics as st
import monthly_sarimax_production as MSX
import monthly_sarima_production as MSA
import monthly_tbats_production as MTB
import monthly_varmax_production as MVX
import monthly_varma_production as MVA
import monthly_holtwinters_production as MHW
import monthly_prophet_production as MPH
import calendar    
from datetime import timedelta
import pickle


data_path = 'C:\\Users\\DN067571\\OneDrive - Cerner Corporation\\Desktop\\Revenue Forecasting\\Datasets\\May_new.csv'
data_path_ff= 'C:\\Users\\DN067571\\OneDrive - Cerner Corporation\\Desktop\\Revenue Forecasting\\Datasets\\May_footfall_new.csv'

########### Get Data #################################################

def get_data(data_path):
    data_charges = pd.read_csv(data_path)
    data_charges=data_charges.set_index("billing_entity")
    data_charges=data_charges.drop(["Adventist Health Physicians Network","Adventist Health and Rideout","Rideout RHC Clinics"]).reset_index()
    return data_charges


############# Check data consistency at daily level ##########################
def data_consistency_date(subset_df,BE):
    dates = subset_df.sort_values('posted_date').reset_index(drop=True)['posted_date'].drop_duplicates()
    start_date = dates.iloc[0]
    end_date = dates.iloc[-1] 
    df_dates =pd.DataFrame(pd.date_range(start_date,end_date),columns=['posted_date'])
    df_dates['posted_date'] = df_dates['posted_date'].apply(str)
    df_dates['billing_entity']=BE
    subset_df1 = pd.merge(df_dates,subset_df,how='outer',on=['posted_date','billing_entity'])
    subset_df1['posted_date'] = pd.to_datetime(subset_df1.posted_date)
    subset_df = subset_df1.sort_values('posted_date').reset_index(drop=True)
    subset_df[['total_charge']] = subset_df[['total_charge']].fillna(0)
    return subset_df


############## Create Overall BE ################################
def create_overall_be(new_df,overall_be,not_include_be_list):
    new_df1 = new_df[~(new_df['billing_entity'].isin(not_include_be_list))]
    new_df1['billing_entity']= overall_be
    full_df=new_df.append(new_df1)
    return full_df


############# Create  Granular Columns ########################
def create_granular_col(df):
    df['posted_date']=pd.to_datetime(df['posted_date'])
    df= df.sort_values('posted_date').reset_index(drop=True)
    df['Month']= df['posted_date'].dt.month
    df['Year']= df['posted_date'].dt.year
    df['Day']= df['posted_date'].dt.day
    df['Weekday']= df['posted_date'].dt.weekday_name
    df['Weekday_num']= df['posted_date'].dt.weekday
    df['MonthYear']= df['Month'].astype('str')+ '-' + df['Year'].astype('str')
    df['WeekNum']= df['posted_date'].dt.week
    return df


################# Remove Partial Data ##########################
def remove_partial_data(modified_df):
    year = np.sort(list(set(modified_df.Year)))
    # Cut off Dates
    cut_off_day_start = 10 #if the Day is greater than cut_off_day_start then we are conisdering it is partial data
    cut_off_day_end = 28   #if the Day is greater than cut_off_day_end then we are conisdering it is partial data
    
    #Subset data with respect to year
    start_year_data = modified_df[modified_df.Year == year[0]].sort_values('posted_date').reset_index(drop=True) # start year data
    end_year_data = modified_df[modified_df.Year == year[-1]].sort_values('posted_date').reset_index(drop=True) # end year data
    
    start_date = start_year_data.posted_date.iloc[0]
    if (len(start_year_data.Month.unique())>1) & (start_date.month !=12):
        if start_date.day > cut_off_day_start:
            start_date = start_date+pd.offsets.MonthBegin()
    else:
        if start_date.day > cut_off_day_start:
            start_date = start_date+pd.offsets.MonthBegin()
        
    end_date = end_year_data.posted_date.iloc[-1]
    
    if end_date.month !=1:
        if end_date.day < cut_off_day_end:
            end_date = end_date-pd.offsets.MonthEnd()
    else:
        if end_date.day < cut_off_day_end:
            end_date = end_date-pd.offsets.MonthEnd()
       
    return start_date,end_date


def subset_data(full_df):
    df_date = pd.DataFrame()
    new_full_df = pd.DataFrame()
    all_be_list = list(sorted(set(full_df.billing_entity)))
    for i in all_be_list:
        modified_df = full_df[full_df.billing_entity==i]
        start_date,end_date = remove_partial_data(modified_df)    
        mask = (modified_df['posted_date'] >= start_date) & (modified_df['posted_date'] <= end_date)
        subset_df= modified_df.loc[mask]
        subset_df = subset_df.sort_values(by='posted_date').reset_index(drop=True)
        if len(subset_df)!=0:
            df_date = df_date.append(pd.DataFrame([i,subset_df.iloc[0]['posted_date'],subset_df.iloc[-1]['posted_date']]).T)
            new_full_df = new_full_df.append(subset_df)
    df_date.columns=['billing_entity','start_date','end_date']
    df_date.reset_index(drop=True,inplace=True)
    new_full_df = new_full_df.sort_values(by='posted_date').reset_index(drop=True)
    return new_full_df,df_date
    
##################### Finding the type of BE ###########################     
    
def be_type(new_full_df, df_date):
    df = pd.DataFrame()
    df = new_full_df.groupby(['billing_entity', 'MonthYear'])['total_charge'].sum().reset_index()
    df = pd.DataFrame(df.groupby('billing_entity')['total_charge'].mean())
    df.drop('All Adventist W', axis=0, inplace=True)
    df.rename(columns={'total_charge': 'monthly_sum'}, inplace=True)
    df.reset_index(inplace=True)
    
    df1 = pd.DataFrame(df.describe())
    df1.columns = ['monthly_sum_desc']
    df1.reset_index(inplace=True)

    all_be_monthly_res = df1.values.tolist()
    
    df_date['type_be'] = ""
    all_be_list = list(sorted(set(df.billing_entity)))   
    for i in all_be_list:
        subset_sum = df[df.billing_entity == i]['monthly_sum'].values[0]
        # If sum of be greater than q3 then large
        if subset_sum >= all_be_monthly_res[6][1]:
            df_date.loc[df_date['billing_entity'] == i, ['type_be']] = 'Large'
        # If sum of be between q2 & q3 then medium
        elif all_be_monthly_res[4][1] < subset_sum <= all_be_monthly_res[6][1]:
             df_date.loc[df_date['billing_entity'] == i, ['type_be']] = 'Medium'
        # If sum of be less than q2 then small
        else:
             df_date.loc[df_date['billing_entity'] == i, ['type_be']] = 'Small'
             
             
    df_date.sort_values('type_be', inplace=True)
    df_date.reset_index(inplace=True)
    df_date.drop("index",axis=1,inplace=True) 
    return df_date
             

#################### Trnasformation (Monthly/Weekly) Based on User's choice ###############
def transformation(subset_df,trans_str,i): 
    transformed_sum=subset_df.resample(trans_str).agg({'total_charge': 'sum','footfall': 'sum'})
    transformed_sum['billing_entity']=i
    return transformed_sum
   
    
    
def get_monthly_agg(new_full_df,choice):
    transformed_data = pd.DataFrame()
    for i in new_full_df.billing_entity.unique():   
        subset_df = new_full_df[new_full_df.billing_entity==i][['posted_date','total_charge','footfall']]
        subset_df.set_index('posted_date',inplace=True)
        trans_str=''
        if choice == "Monthly":
            trans_str='M'
        elif choice == "Weekly":
            trans_str='W'
        transformed_sum = transformation(subset_df,trans_str,i)
        transformed_data = transformed_data.append(transformed_sum)
    return transformed_data


################## Finding the BE with missing recent months & Count of months missing #################
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month 


def recent_months_missing(df_date):
    recent = df_date.end_date.max()
    months_miss = []
    for i in df_date.end_date:
        a=diff_month(recent,i)
        months_miss.append(a)
    df_date['months_missing']=months_miss
    return df_date
    

########################Outlier Imputation #######################
def outlier_value(subset_df):
    q1 = subset_df.quantile(0.25).values
    q3 = subset_df.quantile(0.75).values
    iqr = q3 - q1
    lower_bound=q1 - 1.5 * iqr
    upper_bound=q3 + 1.5 * iqr
    subset_df.reset_index()
    index_value=subset_df.copy()
    index_value["Flag"]=""
    index_value["Flag"]=[ 1 if (x < lower_bound) | (x > upper_bound) else 0 for x in index_value["total_charge"] ]
    return index_value

def sliding_window_outlier_detection(outlier_data):
    outlier_data=outlier_data.reset_index()
    flag_count=pd.DataFrame(columns=["flag","count","occurance"])
    flag_count["flag"]=outlier_data.posted_date
    flag_count=flag_count.set_index("flag")
    flag_count["count"]=0
    flag_count["occurance"]=0
    flag_count["percent"]=0
    
    slider_size=12
    for i in range(len(outlier_data)):
        flaged_dates=[]
        end = i + slider_size
        if end > len(outlier_data):
            break
        index_value=outlier_value(outlier_data[["posted_date","total_charge"]][i:end])
        for x in index_value.posted_date:
            flag_count['occurance'].loc[x]+=1
        flaged_dates=list(index_value[index_value["Flag"] == 1].posted_date)
        for j in flaged_dates:
            flag_count['count'].loc[j]+=1
            
    flag_count["percent"]= flag_count["count"]/flag_count["occurance"]* 100  
    flag_count["accepted_flag"]=[ 1 if (flag_count["percent"].loc[x]==100 and flag_count["occurance"].loc[x]>6) else 0 for x in flag_count.index]  
    return flag_count

  
def sliding_window_outlier_imputation(outlier_data,flag_count):
    outlier_data=outlier_data.reset_index()
    flag_count=flag_count.reset_index()
    for i in flag_count.index:
        if flag_count["accepted_flag"].loc[i]==1:
            print(i)
            a=[]
            b=[]
            c=1
            d=1
            for p in range(1,len(flag_count)):
                if c==7:
                    break
                if flag_count["accepted_flag"][i-p] == 0:
                    a.append(flag_count["flag"][i-p])
                    c=c+1
                else:
                    continue
            for q in range(1,len(flag_count)):
                if d==7:
                    break
                if flag_count["accepted_flag"][i+q] == 0:
                    b.append(flag_count["flag"][i+q])
                    d=d+1
                else:
                    continue  
            dates=[]
            dates=a+b
            median_values=[]
            for x in dates:
                median_values.append(outlier_data[outlier_data.posted_date == x]["total_charge"].values)
                median_charge=st.median(median_values)

            outlier_data["total_charge"].loc[i]=median_charge
        else:
            continue
    outlier_data=outlier_data.set_index("posted_date")
    return outlier_data
              
    
################# Negative charges count ##########################
    
def negative_charges_count(data_agg,df_date):
    neg_charges_be=data_agg[data_agg["total_charge"]<0]
    neg_be=list(neg_charges_be.billing_entity)
    counter=collections.Counter(neg_be)
    df_date["Count_neg"]=''
    df_date["Count_neg"]=[ counter[x] if x in counter.keys()  else 0 for x in df_date.billing_entity ]
    return df_date


############################ Make extended holidays ############################
def make_extended(new_df_monthly,us_holidays):
    new_df_monthly['extended']=0
    for i in new_df_monthly.index:
        if i in list(us_holidays.date):
            if i.weekday()==1:
                j=i-pd.tseries.offsets.Day(1)
                new_df_monthly['extended'].loc[j]=1
            elif i.weekday()==3 :
                j=i+pd.tseries.offsets.Day(1)
                new_df_monthly['extended'].loc[j]=1
    return new_df_monthly
    

############################ Fetch First Date #######################
def fetch_start_date(new_data):
    start_date=new_data.posted_date[0]
    while True:
        if (new_data[new_data.posted_date == start_date]['total_charge'].values[0] ==0):
            start_date = start_date+pd.tseries.offsets.Day(1)
        else:
            break
    return start_date


############################## Holidays Variable ##################################

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
    
def holidays_monthly(new_full_df,us_holidays,i):    
    new_data=new_full_df[new_full_df.billing_entity == i]
    new_data=new_data[["posted_date","total_charge","footfall"]]
    
    last_month = new_data["posted_date"].loc[new_data.index[-1]].month
    last_year = new_data["posted_date"].loc[new_data.index[-1]].year
    last_day=calendar.monthrange(last_year,last_month)[1]
    date1=datetime(last_year,last_month,last_day)
    date2=date1+relativedelta(months=3)
    dates_list=[]
    for dt in daterange(date1, date2):
        dates_list.append(dt.strftime("%Y-%m-%d"))
    new_data=new_data.append(pd.DataFrame(dates_list,columns=["posted_date"]),ignore_index=True)
    new_data=create_granular_col(new_data)
    
    new_data=new_data.set_index("posted_date")
    new_data = new_data.resample("D").agg({'total_charge': 'sum','footfall':'sum'})
    new_data=new_data.reset_index()
    start_date=fetch_start_date(new_data)    
    mask = new_data['posted_date'] >= start_date
    new_data= new_data.loc[mask]
    new_data = new_data.sort_values(by='posted_date').reset_index(drop=True)

    new_data=new_data.set_index("posted_date")
    
    new_data["minor_holiday"]=[1 if (val in list(us_holidays.date) and us_holidays[us_holidays.date==val]["flag"].values[0]==1) else 0 for val in new_data.index] 
    new_data["major_holiday"]=[1 if (val in list(us_holidays.date) and us_holidays[us_holidays.date==val]["flag1"].values[0]==1) else 0 for val in new_data.index]
    new_data["observed_holiday"]=[1 if (val in list(us_holidays.date) and us_holidays[us_holidays.date==val]["flag2"].values[0]==1) else 0 for val in new_data.index]
    new_data=make_extended(new_data,us_holidays)
    new_data['count_weekend']=[1 if (val==5 or val==6) else 0 for val in list(new_data.index.weekday)]
    new_data1=new_data.resample('M').agg({'total_charge': 'sum','footfall':'sum', 'minor_holiday': 'sum', 'major_holiday': 'sum', 
                               'observed_holiday': 'sum', 'extended': 'sum' ,'count_weekend': 'sum'})
    new_data1["billing_entity"]=i
    return new_data1
    

def all_model_combined(new_df_monthly,best_model_name,be_name):

    best_model_res=[]
    train=[]
    val=[]
    test=[]

    outlier_data,test = new_df_monthly[:-3], new_df_monthly[-3:]
    flag_count=sliding_window_outlier_detection(outlier_data)
    if flag_count['accepted_flag'].values.sum()>0:
        while(flag_count['accepted_flag'].values.sum()>0):
            flag_count=sliding_window_outlier_detection(outlier_data)
            outlier_data=sliding_window_outlier_imputation(outlier_data,flag_count)
        train,val=outlier_data[:-3],outlier_data[-3:]
    else:
        train,val=outlier_data[:-3],outlier_data[-3:]
        
    if best_model_name =="SARIMA":
        train_mape_sarima,val_mape_sarima,test_dates,fitted_mean_sarima,fitted_mean_down,fitted_mean_up=MSA.sarima_model(train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name)
        best_model_res.append([be_name,train_mape_sarima,val_mape_sarima,test_dates,fitted_mean_sarima,fitted_mean_down,fitted_mean_up])
   
    if best_model_name =="SARIMAX":    
        train_mape_sarimax,val_mape_sarimax,test_dates,fitted_mean_sarimax,fitted_mean_down,fitted_mean_up=MSX.sarimax_model(train,val,test,be_name)
        best_model_res.append([be_name,train_mape_sarimax,val_mape_sarimax,test_dates,fitted_mean_sarimax,fitted_mean_down,fitted_mean_up])
         
    if best_model_name =="TBATS":     
        train_mape_tbats,val_mape_tbats,test_dates,fitted_mean_tbats,fitted_mean_down,fitted_mean_up=MTB.monthly_tbats(train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name)
        best_model_res.append([be_name,train_mape_tbats,val_mape_tbats,test_dates,fitted_mean_tbats,fitted_mean_down,fitted_mean_up])
    
    if best_model_name =="VARMAX": 
        train_mape_varmax,val_mape_varmax,test_dates,fitted_mean_varmax,fitted_mean_down,fitted_mean_up=MVX.varmax_model(train,val,test,be_name)
        best_model_res.append([be_name,train_mape_varmax,val_mape_varmax,test_dates,fitted_mean_varmax,fitted_mean_down,fitted_mean_up])
        
    if best_model_name =="VARMA":     
        train_mape_varma,val_mape_varma,test_dates,fitted_mean_varma,fitted_mean_down,fitted_mean_up=MVA.varma_model(train,val,test,be_name)
        best_model_res.append([be_name,train_mape_varma,val_mape_varma,test_dates,fitted_mean_varma,fitted_mean_down,fitted_mean_up])
             
    if best_model_name =="Holt Winter's":     
        train_mape_hol_wins,val_mape_hol_wins,test_dates,fitted_mean_hol_wins,fitted_mean_down,fitted_mean_up=MHW.monthly_holtswinter(train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name)
        best_model_res.append([be_name,train_mape_hol_wins,val_mape_hol_wins,test_dates,fitted_mean_hol_wins,fitted_mean_down,fitted_mean_up])
    
    if best_model_name =="PROPHET":
        train_mape_prophet,val_mape_prophet,test_dates,fitted_mean_prophet,fitted_mean_down,fitted_mean_up=MPH.monthly_prophet(train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name)
        best_model_res.append([be_name,train_mape_prophet,val_mape_prophet,test_dates,fitted_mean_prophet,fitted_mean_down,fitted_mean_up])
    
            
    return best_model_res


def report_generation1(new_df_monthly,best_model_res,timestampStr,be_name):
    x1=new_df_monthly[["total_charge"]][:-3].reset_index()
    x1["prediction"]='FALSE'
    x1['posted_date']=[i.strftime("%d-%b-%Y") for i in x1['posted_date']]
    
    x2=pd.DataFrame(best_model_res['test_predicted'][0][:],timestampStr)
    x2["prediction"]='TRUE'
    x2["lower95"]=best_model_res.lower_conf[0][:]
    x2["upper95"]=best_model_res.upper_conf[0][:]
    
    x3 = x2.reset_index()
    x3=x3.rename(columns = {"index": "posted_date", 0:"total_charge"})
    
    x4=x1.append(x3).reset_index(drop=True)
    
    x5=pd.DataFrame(best_model_res.upper_conf[0][:],best_model_res['test_predicted'][0][:])
    x5=x4.replace(np.nan, '', regex=True)
    x5["billing_entity"]=be_name
    
    cols = x5.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    
    x6=x5[cols]
    l=x6[x6["prediction"]=="TRUE"].index[0]
    x6['total_charge']=['{:0,.0f}'.format(i) for i in x6['total_charge']]
    x6.loc[l-0.5]=[x6.loc[l-1].billing_entity,x6.loc[l-1].total_charge,x6.loc[l-1].posted_date,'TRUE',x6.loc[l-1].total_charge,x6.loc[l-1].total_charge]
    x6=x6.sort_index().reset_index(drop=True)
    
    x6=x6[['billing_entity','posted_date','total_charge','prediction','lower95','upper95']]
    return x6


def report_generation2(best_model_res,be_name):
    directory ="C:\\Users\\DN067571\\OneDrive - Cerner Corporation\\Desktop\\Revenue Forecasting\\PickleFiles\\"
    filename1 = "final_report2_3.pickle"
    with open(directory + filename1, 'rb') as f:
        x1_pickle = pickle.load(f)
    x1_pickle=x1_pickle.reset_index(drop=True)
    x1_pickle=x1_pickle[['billing_entity','posted_date','total_charge','prediction','lower95','upper95','prediction_date','prediction_type']].reset_index(drop=True)
    timestampStr = best_model_res["test_dates"][0][:].strftime("%d-%b-%Y")
    x2=pd.DataFrame(best_model_res['test_predicted'][0][:],timestampStr)
    x2['billing_entity']=be_name
    x2["prediction"]='TRUE'
    x2["lower95"]=best_model_res.lower_conf[0][:]
    x2["upper95"]=best_model_res.upper_conf[0][:]
    date=best_model_res["test_dates"][0][0]-relativedelta(months=1)
    last_day=calendar.monthrange(date.year,date.month)[1]
    final_date=pd.to_datetime(str(date.year)+str('-')+str(date.month)+str('-')+str(last_day))
    x2['prediction_date']=final_date.strftime("%d-%b-%Y")
    x2['prediction_type']=['1 month','2 month','3 month']
    x2=x2.rename(columns = {"index": "posted_date", 0:"total_charge"})
    x2['total_charge']=['{:0,.0f}'.format(i) for i in x2['total_charge']]
    x2=x2.reset_index()
    x2=x2.rename(columns = {"index": "posted_date"})
    x2=x2[['billing_entity','posted_date','total_charge','prediction','lower95','upper95','prediction_date','prediction_type']].reset_index(drop=True)
    x3=x1_pickle.append(x2)
    with open(directory+filename1, 'wb') as handle:
       pickle.dump(x3, handle)
    return x3    

def report_generation_merge(be_name,new_df_monthly,x3):
    x1=new_df_monthly[["total_charge"]][:-3].reset_index()
    x1['billing_entity']=be_name
    x1['total_charge']=['{:0,.0f}'.format(i) for i in x1['total_charge']]
    x1["prediction"]='FALSE'
    x1['posted_date']=[i.strftime("%d-%b-%Y") for i in x1['posted_date']]
    x1['lower95']=''
    x1['upper95']=''
    x1['prediction_date']=''
    x1['prediction_type']=''
    x1=x1[['billing_entity','posted_date','total_charge','prediction','lower95','upper95','prediction_date','prediction_type']].reset_index(drop=True)
    x3=x3[x3.billing_entity==be_name]
    x4=x1.append(x3).reset_index(drop=True)
    x4=x4.replace(np.nan, '', regex=True)
    return x4


###################### Main Function ###########################################
def main_func(data_path,data_path_ff):   
    data = get_data(data_path)
    all_be_list = list(sorted(set(data.billing_entity)))
    new_df = pd.DataFrame()
    for i in all_be_list:
        a = data_consistency_date(data[data.billing_entity==i],i)
        new_df = new_df.append(a)
    ff_data=pd.read_csv(data_path_ff)
    ff_data.columns=['billing_entity','posted_date','encounter_class','financial_class','footfall']
    ff_data=ff_data.set_index("billing_entity")
    ff_data['posted_date'] = pd.to_datetime(ff_data.posted_date)
    ff_data=ff_data.dropna().reset_index()
    df=pd.merge(new_df,ff_data,how='outer',on=['posted_date','billing_entity','encounter_class','financial_class'])
    df.footfall.fillna(0,inplace=True)
    df.total_charge.fillna(0,inplace=True)
    not_include_be_list=["Feather River Hospital"]
    overall_be = "All Adventist W"
    full_df = create_overall_be(df,overall_be, not_include_be_list)
    full_df = create_granular_col(full_df)
    new_full_df, df_date = subset_data(full_df)
    
   ############ FInding the count of months in each billing entity##############
    full_data = new_full_df.groupby(['billing_entity','MonthYear'])
    full_data = full_data.MonthYear.first().groupby(level=0).size().reset_index()
    full_data.columns = ['billing_entity','count_of_months']
    full_data.sort_values(['count_of_months','billing_entity'],ascending=False,inplace=True)
    full_data = full_data.reset_index(drop=True)
    
    df_date = pd.merge(df_date,full_data,how='outer',on='billing_entity')
    df_date.sort_values(['count_of_months','billing_entity'],ascending=False,inplace=True)
    df_date['valid'] = df_date['count_of_months']>=9
    choice="Monthly" #### choice="Weekly"
    data_agg = get_monthly_agg(new_full_df,choice)
    df_date = be_type(new_full_df,df_date)
    df_date = recent_months_missing(df_date)
    df_date = negative_charges_count(data_agg, df_date)
    
    new_full_df=new_full_df.set_index("billing_entity")
    new_full_df.drop(list(df_date[df_date.count_of_months < 9]['billing_entity']), axis=0, inplace=True)
    new_full_df.drop(list(df_date[df_date.months_missing > 0]['billing_entity']), axis=0, inplace=True)
    new_full_df.reset_index(inplace=True)
    
    us_holidays=[]
    for date in holidays.UnitedStates(years=[2015,2016,2017,2018,2019,2020,2021]).items():
        us_holidays.append([str(date[0]),date[1]])
    us_holidays=pd.DataFrame(us_holidays,columns=['date','holiday'])
    us_holidays['date']=pd.to_datetime(us_holidays["date"])
    us_holidays.holiday=us_holidays.holiday.astype(str)
    us_holidays['flag']=0
    us_holidays['flag']=[1 if (i=='Martin Luther King, Jr. Day' or i=="Washington's Birthday" or i=='Columbus Day' or i=='Veterans Day') else 0 for i in us_holidays.holiday.astype("str").values]
    us_holidays['flag1']=0
    us_holidays['flag1']=[1 if (i=="New Year's Day" or i=="Christmas Day" or i=='Thanksgiving' or i=='Memorial Day' or i=='Labor Day' or i=='Independence Day') else 0 for i in us_holidays.holiday.astype("str").values]
    us_holidays['flag2']=0
    us_holidays['flag2']=[1 if (i.endswith("(Observed)")) else 0 for i in us_holidays.holiday.astype("str").values]

    dir="C:\\Users\\DN067571\\OneDrive - Cerner Corporation\\Desktop\\Revenue Forecasting\\Final Walk Forward Code\\"
    best_model_name=pd.read_csv(dir+"best_model_may.csv")
    
       
    final_report1=pd.DataFrame()
    final_report2=pd.DataFrame()

    for k in tqdm(best_model_name.index):
        be_name=best_model_name["be_name"].loc[k]
        best_model=best_model_name["best_model"].loc[k]
        print(k,be_name,best_model)
        new_df_monthly=holidays_monthly(new_full_df,us_holidays,be_name)
        best_model_res=all_model_combined(new_df_monthly,best_model,be_name)        
        best_model_res=pd.DataFrame(best_model_res,columns=['billing_entity','train_mape','val_mape','test_dates','test_predicted','lower_conf','upper_conf'])
        timestampStr = best_model_res["test_dates"][0][:].strftime("%d-%b-%Y")
        best_model_res['lower_conf'][0][:]=['{:0,.0f}'.format(i) for i in best_model_res['lower_conf'][0][:]]
        best_model_res['upper_conf'][0][:]=['{:0,.0f}'.format(i) for i in best_model_res['upper_conf'][0][:]]

        report1=report_generation1(new_df_monthly,best_model_res,timestampStr,be_name)
        report2=report_generation2(best_model_res,be_name)
        final_report=report_generation_merge(be_name,new_df_monthly,report2)
        
        final_report1=final_report1.append(report1)
        final_report2=final_report2.append(final_report)
    
    final_report1=final_report1.set_index("billing_entity")
    final_report1.to_csv("final_report1_may.csv")
    final_report2=final_report2.set_index("billing_entity")
    final_report2.to_csv("final_report2_may.csv")
    
   


