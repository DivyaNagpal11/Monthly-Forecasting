# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:42:11 2020

@author: DN067571
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:47:30 2020

@author: DN067571
"""

import pandas as pd
import numpy as np
#from plotly.offline import plot
import collections
import statistics as st
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')

import monthly_sarimax as MSX
import monthly_sarima as MSA
import monthly_tbats as MTB
import monthly_varmax as MVX
import monthly_varma as MVA
import monthly_holtwinters as MHW
import monthly_prophet as MPH

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
             

##################### Trnasformation (Monthly/Weekly) Based on User's choice ###############
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
    df_date['count_recent_months_missing']=months_miss
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


############################ Make extended holidays #########################################
def make_extended(new_df_monthly,us_holidays):
    new_df_monthly['is_extended']=0
    for i in new_df_monthly.index:
        if i in list(us_holidays.date):
            if i.weekday()==1:
                j=i-pd.tseries.offsets.Day(1)
                new_df_monthly['is_extended'].loc[j]=1
            elif i.weekday()==3 :
                j=i+pd.tseries.offsets.Day(1)
                new_df_monthly['is_extended'].loc[j]=1
    return new_df_monthly
    

############################ Fetch First Date ###############################################
def fetch_start_date(new_data):
    start_date=new_data.posted_date[0]
    while True:
        if (new_data[new_data.posted_date == start_date]['total_charge'].values[0] ==0):
            start_date = start_date+pd.tseries.offsets.Day(1)
        else:
            break
    return start_date


############################## Holidays Variable ###########################################
def holidays_monthly(new_full_df,us_holidays,i):    
    new_data=new_full_df[new_full_df.billing_entity == i]
    new_data=new_data[["posted_date","total_charge","footfall"]]
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
                               'observed_holiday': 'sum', 'is_extended': 'sum' ,'count_weekend': 'sum'})
    return new_data1


def all_model_combined(new_df_monthly, n_steps_in, n_steps_out,be_name,set_no):
    sarima_mape_res=[]
    sarimax_mape_res=[]
    tbats_mape_res=[]
    varmax_mape_res=[]
    varma_mape_res=[]
    holtwinter_mape_res=[]
    prophet_mape_res=[]
    

    new_df_monthly1=new_df_monthly[-(n_steps_in+n_steps_out):]    
    outlier_data,test = new_df_monthly1[:n_steps_in], new_df_monthly1[-n_steps_out:]
    
    flag_count=sliding_window_outlier_detection(outlier_data)
    if flag_count['accepted_flag'].values.sum()>0:
        while(flag_count['accepted_flag'].values.sum()>0):
            flag_count=sliding_window_outlier_detection(outlier_data)
            outlier_data=sliding_window_outlier_imputation(outlier_data,flag_count)
        train,val=outlier_data[:-3],outlier_data[-3:]
    else:
        train,val=outlier_data[:-3],outlier_data[-3:]
    
    c=set_no
    train_mape_sarima,val_mape_sarima,test_mape_sarima,one_month_sarima,two_month_sarima,third_month_sarima,test_set_sarima,fitted_mean_sarima=MSA.sarima_model(train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name,c)
    sarima_mape_res.append([be_name,"SARIMA",c,train_mape_sarima,val_mape_sarima,test_mape_sarima,one_month_sarima,two_month_sarima,third_month_sarima,test_set_sarima,fitted_mean_sarima])
    
    train_mape_sarimax,val_mape_sarimax,test_mape_sarimax,one_month_sarimax,two_month_sarimax,third_month_sarimax,test_set_sarimax,fitted_mean_sarimax=MSX.sarimax_model(train,val,test,be_name,c)
    sarimax_mape_res.append([be_name,"SARIMAX",c,train_mape_sarimax,val_mape_sarimax,test_mape_sarimax,one_month_sarimax,two_month_sarimax,third_month_sarimax,test_set_sarimax,fitted_mean_sarimax])
            
    train_mape_tbats,val_mape_tbats,test_mape_tbats,one_month_tbats,two_month_tbats,third_month_tbats,test_set_tbats,fitted_mean_tbats=MTB.monthly_tbats(train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name,c)
    tbats_mape_res.append([be_name,"TBATS",c,train_mape_tbats,val_mape_tbats,test_mape_tbats,one_month_tbats,two_month_tbats,third_month_tbats,test_set_tbats,fitted_mean_tbats])
    
    train_mape_varmax,val_mape_varmax,test_mape_varmax,one_month_varmax,two_month_varmax,third_month_varmax,test_set_varmax,fitted_mean_varmax=MVX.varmax_model(train,val,test,be_name,c)
    varmax_mape_res.append([be_name,"VARMAX",c,train_mape_varmax,val_mape_varmax,test_mape_varmax,one_month_varmax,two_month_varmax,third_month_varmax,test_set_varmax,fitted_mean_varmax])
    
    train_mape_varma,val_mape_varma,test_mape_varma,one_month_varma,two_month_varma,third_month_varma,test_set_varma,fitted_mean_varma=MVA.varma_model(train,val,test,be_name,c)
    varma_mape_res.append([be_name,"VARMA",c,train_mape_varma,val_mape_varma,test_mape_varma,one_month_varma,two_month_varma,third_month_varma,test_set_varma,fitted_mean_varma])
            
    train_mape_hol_wins,val_mape_hol_wins,test_mape_hol_wins,one_month_hol_wins,two_month_hol_wins,third_month_hol_wins,test_set_hol_wins,fitted_mean_hol_wins=MHW.monthly_holtswinter(train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name,c)
    holtwinter_mape_res.append([be_name,"Holt Winter's",c,train_mape_hol_wins,val_mape_hol_wins,test_mape_hol_wins,one_month_hol_wins,two_month_hol_wins,third_month_hol_wins,test_set_hol_wins,fitted_mean_hol_wins])
    
    train_mape_prophet,val_mape_prophet,test_mape_prophet,one_month_prophet,two_month_prophet,third_month_prophet,test_set_prophet,fitted_mean_prophet=MPH.monthly_prophet(train[["total_charge"]],val[["total_charge"]],test[["total_charge"]],be_name,c)
    prophet_mape_res.append([be_name,"PROPHET",c,train_mape_prophet,val_mape_prophet,test_mape_prophet,one_month_prophet,two_month_prophet,third_month_prophet,test_set_prophet,fitted_mean_prophet])
    
    return sarima_mape_res,sarimax_mape_res,tbats_mape_res,varmax_mape_res,varma_mape_res,holtwinter_mape_res,prophet_mape_res
 