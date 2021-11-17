# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:32:07 2020

@author: DN067571
"""

import monthly_data_preparation_footfall as MDP
import holidays
import pandas as pd


def main(data_path,data_path_ff):
    data = MDP.get_data(data_path)
    all_be_list = list(sorted(set(data.billing_entity)))
    new_df = pd.DataFrame()
    for i in all_be_list:
        a = MDP.data_consistency_date(data[data.billing_entity==i],i)
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
    full_df = MDP.create_overall_be(df,overall_be, not_include_be_list)
    full_df = MDP.create_granular_col(full_df)
    new_full_df, df_date = MDP.subset_data(full_df)
    
       ############ FInding the count of months in each billing entity##############
    full_data = new_full_df.groupby(['billing_entity','MonthYear'])
    full_data = full_data.MonthYear.first().groupby(level=0).size().reset_index()
    full_data.columns = ['billing_entity','count_of_months']
    full_data.sort_values(['count_of_months','billing_entity'],ascending=False,inplace=True)
    full_data = full_data.reset_index(drop=True)
    
    df_date = pd.merge(df_date,full_data,how='outer',on='billing_entity')
    df_date.sort_values(['count_of_months','billing_entity'],ascending=False,inplace=True)

    choice="Monthly" #### choice="Weekly"
    data_agg = MDP.get_monthly_agg(new_full_df,choice)
    df_date = MDP.be_type(new_full_df,df_date)
    df_date = MDP.recent_months_missing(df_date)
    df_date = MDP.negative_charges_count(data_agg, df_date)
    
    new_full_df=new_full_df.set_index("billing_entity")
    new_full_df.drop(list(df_date[df_date.count_of_months < 9]['billing_entity']), axis=0, inplace=True)
    new_full_df.drop(list(df_date[df_date.count_recent_months_missing > 0]['billing_entity']), axis=0, inplace=True)
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
    
#    csv_folder=r"csv_outlier/"    
    data_length=pd.read_csv("data_length_info.csv")
    walk_forward_file=pd.read_csv("walk_forward_file_april.csv")
    for j in new_full_df.billing_entity.unique():
        new_df_monthly=MDP.holidays_monthly(new_full_df,us_holidays,j)
        set_no=walk_forward_file[walk_forward_file.billing_entity==j]["set_no"].max()+1
        n_steps_in=data_length[data_length.billing_entity==j]["n_steps_in"].values[0]
        n_steps_out=data_length[data_length.billing_entity==j]["n_steps_out"].values[0]
        
        sarima_mape_res,sarimax_mape_res,tbats_mape_res,varmax_mape_res,varma_mape_res,holtwinter_mape_res,prophet_mape_res=MDP.all_model_combined(new_df_monthly, n_steps_in, n_steps_out,j,set_no)
        
        sarima_mape_res=pd.DataFrame(sarima_mape_res,columns=['billing_entity','model','set_no','train_mape','val_mape','test_mape','one_month','two_month','third_month','test_actual','test_predicted'])
        sarimax_mape_res=pd.DataFrame(sarimax_mape_res,columns=['billing_entity','model','set_no','train_mape','val_mape','test_mape','one_month','two_month','third_month','test_actual','test_predicted'])
        tbats_mape_res=pd.DataFrame(tbats_mape_res,columns=['billing_entity','model','set_no','train_mape','val_mape','test_mape','one_month','two_month','third_month','test_actual','test_predicted'])
        varmax_mape_res=pd.DataFrame(varmax_mape_res,columns=['billing_entity','model','set_no','train_mape','val_mape','test_mape','one_month','two_month','third_month','test_actual','test_predicted'])
        varma_mape_res=pd.DataFrame(varma_mape_res,columns=['billing_entity','model','set_no','train_mape','val_mape','test_mape','one_month','two_month','third_month','test_actual','test_predicted'])
        holtwinter_mape_res=pd.DataFrame(holtwinter_mape_res,columns=['billing_entity','model','set_no','train_mape','val_mape','test_mape','one_month','two_month','third_month','test_actual','test_predicted'])
        prophet_mape_res=pd.DataFrame(prophet_mape_res,columns=['billing_entity','model','set_no','train_mape','val_mape','test_mape','one_month','two_month','third_month','test_actual','test_predicted'])
        
        mape_res=pd.concat([sarima_mape_res,sarimax_mape_res,tbats_mape_res,varmax_mape_res,varma_mape_res,holtwinter_mape_res,prophet_mape_res])
        walk_forward_file=walk_forward_file.append(mape_res)        
        
    walk_forward_file.to_csv('walk_forward_file_may.csv')    

           
if __name__ == '__main__':
    data_path="May_new.csv"
    data_path_ff= "May_footfall_new.csv"
    main(data_path,data_path_ff)



