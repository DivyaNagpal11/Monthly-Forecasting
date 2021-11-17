import json
import pickle
import holidays
import collections
import numpy as np
import pandas as pd
import statistics as st
from os import path,makedirs,listdir
from sklearn import preprocessing
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from datetime import datetime
import calendar   
from datetime import datetime

class DataPreprocessing:
    
    """
    data_path: specify path of the file
    not_include_be_list: list of Billing Entity that we should ommit
    overall_be: specify name of overall billing entity
    aggregate_choice: specify whether you need transformation Weekly or Monthly by default its daily
    dates: date column
    self.target_value: target column for which we need to predict
    items: granural columns by which we have to do prediction
    eg:- 
    target_dates = 'posted_date'
    target_value = 'total_charge'
    target_items = 'billing_entity'
    """
    
    # STEP1: Data Initialization 
    def __init__(self,data_path,not_include_be_list,overall_be,target_dates,target_value,target_items,target_file,other_target_value,aggregate_choice,minor_holidays,major_holidays,n_periods,wf):
        self.data_path = data_path
        self.target_file = target_file
        self.other_target_value = other_target_value
        self.data = pd.DataFrame()
        self.not_include_be_list = not_include_be_list
        self.overall_be = overall_be
        self.summary_df = pd.DataFrame()
        self.transform_data = pd.DataFrame()
        self.choice = aggregate_choice
        self.target_dates = target_dates
        self.target_value = target_value
        self.target_items = target_items 
        self.minor_holidays = minor_holidays
        self.major_holidays = major_holidays
        self.n_periods = n_periods
        self.walk_forward = wf
    
    #STEP 2: Data Conistency 
    def data_consistency_date(self, data):
        resample_data = data.set_index(pd.to_datetime(data[self.target_dates])).groupby([self.target_items]).resample('D').sum().reset_index()
        return resample_data
    
    
    def combine_data(self):
        DATA = {}
        files = [file.replace(".csv","") for file in listdir(self.data_path)]
        for file in files:            
            #Create data consistency
            DATA[file] = self.data_consistency_date(pd.read_csv(self.data_path+file+'.csv'))
            DATA[file][self.target_dates] = pd.to_datetime(DATA[file][self.target_dates])
            DATA[file].sort_values(self.target_dates).reset_index(drop=True,inplace=True)
#            print("DATA COMBINE",DATA[file],len(DATA[file]))
        df = DATA[self.target_file]
        for file in DATA:
            if file != self.target_file:
                df = pd.merge(df,DATA[file],how='left',on=[self.target_dates,self.target_items])
        df.sort_values(self.target_dates).reset_index(drop=True,inplace=True)
        self.data = df.copy()
        return df
    
    #STEP 3: Create Overall Billing Entity
    def create_overall_be(self, data_update):
        new_df = data_update[~(data_update[self.target_items].isin(self.not_include_be_list))]
        all_advh = new_df.copy()
        all_advh[self.target_items] = self.overall_be 
        full_df = new_df.append(all_advh)
        return full_df
    
    
    #STEP 4: Create Granurlar columns
    def create_granular_col(self, overall_be_data):
        overall_be_data[self.target_dates] = pd.to_datetime(overall_be_data[self.target_dates])
        overall_be_data = overall_be_data.sort_values(self.target_dates).reset_index(drop=True)
        overall_be_data['Month'] = overall_be_data[self.target_dates].dt.month
        overall_be_data['Year'] = overall_be_data[self.target_dates].dt.year
        overall_be_data['Day'] = overall_be_data[self.target_dates].dt.day
        overall_be_data['Weekday'] = overall_be_data[self.target_dates].dt.weekday_name
        overall_be_data['Weekday_num'] = overall_be_data[self.target_dates].dt.weekday
        overall_be_data['MonthYear'] = overall_be_data['Month'].astype('str')+ '-' + overall_be_data['Year'].astype('str')
        overall_be_data['WeekNum'] = overall_be_data[self.target_dates].dt.week
        return overall_be_data


    
    #Update summary file
    def update_summary(self, data, summary_df):
        #Sum of total charges per billing entity
        sum_data  = data.groupby([self.target_items, 'MonthYear'])[self.target_value].sum().reset_index()
        mean_data = pd.DataFrame(sum_data.groupby(self.target_items)[self.target_value].mean())    
         
        
        mean_data.drop(self.overall_be, axis=0, inplace=True)
        mean_data.rename(columns={self.target_value: 'monthly_sum'}, inplace=True)
        mean_data.reset_index(inplace=True)
    

        q25 =  mean_data.quantile(0.25)[0]
        q75 =  mean_data.quantile(0.75)[0]

        
        summary_df['type_be'] = ""
        
        all_be_list = list(mean_data[self.target_items].unique())
        
        for BE in all_be_list:
            subset_sum = mean_data[mean_data[self.target_items] == BE]['monthly_sum'].values[0]
            
            # If sum of be greater than q3 then large
            if subset_sum > q75:
                summary_df.loc[summary_df[self.target_items] == BE, ['type_be']] = 'Large'
            
            # Else if sum of be between q2 & q3 then medium
            elif q25 < subset_sum <= q75:
                summary_df.loc[summary_df[self.target_items] == BE, ['type_be']] = 'Medium'
            # Else: sum of be less than q2 then small
            else:
                summary_df.loc[summary_df[self.target_items] == BE, ['type_be']] = 'Small'
           
        summary_df.sort_values('type_be', inplace=True)
        summary_df.reset_index(inplace = True, drop=True)
        return summary_df
    
    
    #STEP 5: Check for partial data
    def remove_partial_data(self,modified_df,cut_off_day_start = 10 ,cut_off_day_end = 28 ):       
        year = np.sort(list(set(modified_df.Year)))

        start_year_data = modified_df[modified_df.Year == year[0]].sort_values(self.target_dates).reset_index(drop=True) # start year data
        end_year_data = modified_df[modified_df.Year == year[-1]].sort_values(self.target_dates).reset_index(drop=True) # end year data
        
        start_date = start_year_data[self.target_dates].iloc[0]
        if (len(start_year_data.Month.unique())>1) & (start_date.month !=12):
            if start_date.day > cut_off_day_start:
                start_date = start_date+pd.offsets.MonthBegin()
        else:
            start_date = start_date+pd.offsets.MonthBegin()
            
        end_date = end_year_data[self.target_dates].iloc[-1]
        
        if end_date.month !=1:
            if end_date.day < cut_off_day_end:
                end_date = end_date-pd.offsets.MonthEnd()
        else:
            if end_date.day < cut_off_day_end:
                end_date = end_date-pd.offsets.MonthEnd()
        return start_date,end_date
    
    
    def subset_data(self,data):
        summary_df = pd.DataFrame()
        new_full_df = pd.DataFrame()
        all_be_list = list(sorted(set(data[self.target_items])))
        for BE in all_be_list:
            modified_df = data[data[self.target_items]==BE]
            start_date,end_date = self.remove_partial_data(modified_df)    
            mask = (modified_df[self.target_dates] >= start_date) & (modified_df[self.target_dates] <= end_date)
            subset_df = modified_df.loc[mask]
            subset_df = subset_df.sort_values(by=self.target_dates).reset_index(drop=True)
            if len(subset_df)!=0:
                summary_df = summary_df.append(pd.DataFrame([BE,subset_df.iloc[0][self.target_dates],subset_df.iloc[-1][self.target_dates]]).T)
                new_full_df = new_full_df.append(subset_df)
        summary_df.columns=[self.target_items,'start_date','end_date']
        summary_df.reset_index(drop=True,inplace=True)
#        current_date = summary_df.end_date.max()
#        summary_df['months_missing'] = summary_df.end_date.apply(lambda row: (current_date.year-row.year)*12 + current_date.month-row.month)
        new_full_df = new_full_df.sort_values(by=self.target_dates).reset_index(drop=True)
        return new_full_df,summary_df
    
    
    def transformation(self,subset_df,trans_str,i): 
        transformed_sum=subset_df.resample(trans_str).agg({self.target_value: 'sum',self.other_target_value: 'sum'})
        transformed_sum[self.target_items]=i
        return transformed_sum
   
    
    def aggregate_data(self,data):
        transformed_data = pd.DataFrame()
        for i in data[self.target_items].unique():   
            subset_df = data[data[self.target_items]==i][[self.target_dates,self.target_value,self.other_target_value]]
            subset_df.set_index(self.target_dates,inplace=True)
            trans_str=''
            if self.choice == "Monthly":
                trans_str='M'
            elif self.choice == "Daily":
                trans_str='D'
            transformed_sum = self.transformation(subset_df,trans_str,i)
            transformed_data = transformed_data.append(transformed_sum)
        transformed_data.to_csv("trans.csv")
        return transformed_data


################## Finding the BE with missing recent months & Count of months missing #################
    def diff_month(self,d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month 
    
    
    def recent_months_missing(self,df_date):
        recent = df_date.end_date.max()
        months_miss = []
        for i in df_date.end_date:
            a=self.diff_month(recent,i)
            months_miss.append(a)
        df_date['missing_recent_months_count']=months_miss
        return df_date


    # NEG CHARGES COUNT
    def negative_charges_count(self, data_agg, data):
        neg_charges_be = data_agg[data_agg[self.target_value]<0]
        neg_be = list(neg_charges_be[self.target_items])
        counter = collections.Counter(neg_be)
        data["Count_neg"] = ''
        data["Count_neg"] = [counter[x] if x in counter.keys()  else 0 for x in data[self.target_items]]
        return data


    #US Holidays
    def usa_holidays(self):
        us_holidays=[]
        for date in holidays.UnitedStates(years=[2017,2018,2019,2020]).items():
            us_holidays.append([str(date[0]),date[1]])
        us_holidays = pd.DataFrame(us_holidays,columns=[self.target_dates,'holiday'])
        us_holidays[self.target_dates] = pd.to_datetime(us_holidays[self.target_dates])
        us_holidays.holiday = us_holidays.holiday.astype(str)
        us_holidays['flag'] = 0
        us_holidays['flag'] = [1 if (i in self.minor_holidays) else 0 for i in us_holidays.holiday.astype("str").values]
        us_holidays['flag1'] = 0
        us_holidays['flag1'] = [1 if (i in self.major_holidays) else 0 for i in us_holidays.holiday.astype("str").values]
        us_holidays['flag2'] = 0
        us_holidays['flag2'] = [1 if (i.endswith("(Observed)")) else 0 for i in us_holidays.holiday.astype("str").values]
        return us_holidays
    
    
    ############################ Fetch First Date ###############################################
    def fetch_start_date(self,new_data):
        start_date=new_data.posted_date[0]
        while True:
            if (new_data[new_data.posted_date == start_date][self.target_value].values[0] ==0):
                start_date = start_date+pd.tseries.offsets.Day(1)
            else:
                break
        return start_date


    ################################# MAKE EXTENDED HOLIDAYS ###########################################3
    def make_extended(self, new_df_monthly, us_holidays):
        new_df_monthly['is_extended'] = 0
        for i in new_df_monthly.index:
            if i in list(us_holidays[self.target_dates]):
                if i.weekday()==1:
                    j = i-pd.tseries.offsets.Day(1)
                    new_df_monthly['is_extended'].loc[j] = 1
                elif i.weekday()==3 :
                    j = i+pd.tseries.offsets.Day(1)
                    new_df_monthly['is_extended'].loc[j] = 1
        return new_df_monthly
    
    
    def holidays_monthly_wf(self,new_full_df,us_holidays,i):    
        new_data=new_full_df[new_full_df[self.target_items] == i]
        new_data=new_data[[self.target_dates,self.target_value,self.other_target_value]]
        new_data=new_data.set_index(self.target_dates)
        new_data = new_data.resample("D").agg({self.target_value: 'sum',self.other_target_value:'sum'})
        new_data=new_data.reset_index()
        start_date=self.fetch_start_date(new_data)    
        mask = new_data[self.target_dates] >= start_date
        new_data= new_data.loc[mask]
        new_data = new_data.sort_values(by=self.target_dates).reset_index(drop=True)
        new_data=new_data.set_index(self.target_dates)
        
        new_data["minor_holiday"]=[1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag"].values[0]==1) else 0 for val in new_data.index] 
        new_data["major_holiday"]=[1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag1"].values[0]==1) else 0 for val in new_data.index]
        new_data["observed_holiday"]=[1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag2"].values[0]==1) else 0 for val in new_data.index]
        new_data=self.make_extended(new_data,us_holidays)
        new_data['count_weekend']=[1 if (val==5 or val==6) else 0 for val in list(new_data.index.weekday)]
        new_data1=new_data.resample('M').agg({self.target_value: 'sum',self.other_target_value:'sum', 'minor_holiday': 'sum', 'major_holiday': 'sum',
                                   'observed_holiday': 'sum', 'is_extended': 'sum' ,'count_weekend': 'sum'})
        return new_data1
    
        
    def daterange(self,date1, date2):
        for n in range(int ((date2 - date1).days)+1):
            yield date1 + timedelta(n)
        
    def holidays_monthly_pd(self,new_full_df,us_holidays,i):    
        new_data=new_full_df[new_full_df[self.target_items] == i]
        new_data=new_data[[self.target_dates,self.target_value,self.other_target_value]]
        
        last_month = new_data[self.target_dates].loc[new_data.index[-1]].month
        last_year = new_data[self.target_dates].loc[new_data.index[-1]].year
        last_day=calendar.monthrange(last_year,last_month)[1]
        date1=datetime(last_year,last_month,last_day)
        date2=date1+relativedelta(months=self.n_periods)
        dates_list=[]
        for dt in self.daterange(date1, date2):
            dates_list.append(dt.strftime("%Y-%m-%d"))
        new_data=new_data.append(pd.DataFrame(dates_list,columns=[self.target_dates]),ignore_index=True)
        new_data=self.create_granular_col(new_data)
        
        new_data=new_data.set_index(self.target_dates)
        new_data = new_data.resample("D").agg({self.target_value:'sum',self.other_target_value:'sum'})
        new_data=new_data.reset_index()
        start_date=self.fetch_start_date(new_data)    
        mask = new_data[self.target_dates] >= start_date
        new_data= new_data.loc[mask]
        new_data = new_data.sort_values(by=self.target_dates).reset_index(drop=True)
    
        new_data=new_data.set_index(self.target_dates)
        
        new_data["minor_holiday"]=[1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag"].values[0]==1) else 0 for val in new_data.index] 
        new_data["major_holiday"]=[1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag1"].values[0]==1) else 0 for val in new_data.index]
        new_data["observed_holiday"]=[1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag2"].values[0]==1) else 0 for val in new_data.index]
        new_data=self.make_extended(new_data,us_holidays)
        new_data['count_weekend']=[1 if (val==5 or val==6) else 0 for val in list(new_data.index.weekday)]
        new_data1=new_data.resample('M').agg({self.target_value: 'sum',self.other_target_value:'sum', 'minor_holiday': 'sum', 'major_holiday': 'sum','observed_holiday': 'sum', 'extended': 'sum' ,'count_weekend': 'sum'})
        new_data1[self.target_items]=i
        return new_data1
    
    ########################Outlier Imputation #######################
    def outlier_value(self,subset_df):
        q1 = subset_df.quantile(0.25).values
        q3 = subset_df.quantile(0.75).values
        iqr = q3 - q1
        lower_bound=q1 - 1.5 * iqr
        upper_bound=q3 + 1.5 * iqr
        subset_df.reset_index()
        index_value=subset_df.copy()
        index_value["Flag"]=""
        index_value["Flag"]=[ 1 if (x < lower_bound) | (x > upper_bound) else 0 for x in index_value[self.target_value] ]
        return index_value
    
    
    def sliding_window_outlier_detection(self,outlier_data):
        outlier_data=outlier_data.reset_index()
        flag_count=pd.DataFrame(columns=["flag","count","occurance"])
        flag_count["flag"]=outlier_data[self.target_dates]
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
            index_value=self.outlier_value(outlier_data[[self.target_dates,self.target_value]][i:end])
            for x in index_value.posted_date:
                flag_count['occurance'].loc[x]+=1
            flaged_dates=list(index_value[index_value["Flag"] == 1][self.target_dates])
            for j in flaged_dates:
                flag_count['count'].loc[j]+=1
                
        flag_count["percent"]= flag_count["count"]/flag_count["occurance"]* 100  
        flag_count["accepted_flag"]=[ 1 if (flag_count["percent"].loc[x]==100 and flag_count["occurance"].loc[x]>6) else 0 for x in flag_count.index]  
        return flag_count
    
      
    def sliding_window_outlier_imputation(self,outlier_data,flag_count):
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
                    median_values.append(outlier_data[outlier_data[self.target_dates] == x][self.target_value].values)
                    median_charge=st.median(median_values)
    
                outlier_data[self.target_value].loc[i]=median_charge
            else:
                continue
        outlier_data=outlier_data.set_index(self.target_dates)
        return outlier_data
    
    
    def all_model_combined_wf(self,new_df_monthly, n_steps_in, n_steps_out,be_name):     
        new_df_monthly1=new_df_monthly[-(n_steps_in+n_steps_out):]   
        
        outlier_data,test = new_df_monthly1[:n_steps_in], new_df_monthly1[-n_steps_out:]
        
        flag_count=self.sliding_window_outlier_detection(outlier_data)
        if flag_count['accepted_flag'].values.sum()>0:
            while(flag_count['accepted_flag'].values.sum()>0):
                flag_count=self.sliding_window_outlier_detection(outlier_data)
                outlier_data=self.sliding_window_outlier_imputation(outlier_data,flag_count)
            train,val=outlier_data[:-int(self.n_periods)],outlier_data[-int(self.n_periods):]
        else:
            train,val=outlier_data[:-int(self.n_periods)],outlier_data[-int(self.n_periods):]
            
        BE_df=pd.concat([train,val,test])    
        return  BE_df
      

    def main_method_wf(self, all_be_df, us_holidays):
        print("in main_wf")
        data_length=pd.read_csv("data/data_length_info.csv")
        final_df = pd.DataFrame()
        for BE in all_be_df[self.target_items].unique().tolist():
            print(BE)
            new_df_monthly=self.holidays_monthly_wf(all_be_df,us_holidays,BE)
            n_steps_in=data_length[data_length[self.target_items]==BE]["n_steps_in"].values[0]
            n_steps_out=data_length[data_length[self.target_items]==BE]["n_steps_out"].values[0]
            BE_df=self.all_model_combined_wf(new_df_monthly, n_steps_in, n_steps_out,BE)
            BE_df[self.target_items]=BE
            final_df = final_df.append(BE_df)
        return final_df
  
    
    def all_model_combined_pd(self,new_df_monthly,be_name):
        train=[]
        val=[]
        test=[]
    
        outlier_data,test = new_df_monthly[:-self.n_periods], new_df_monthly[-self.n_periods:]
        flag_count=self.sliding_window_outlier_detection(outlier_data)
        if flag_count['accepted_flag'].values.sum()>0:
            while(flag_count['accepted_flag'].values.sum()>0):
                flag_count=self.sliding_window_outlier_detection(outlier_data)
                outlier_data=self.sliding_window_outlier_imputation(outlier_data,flag_count)
            train,val=outlier_data[:-self.n_periods],outlier_data[-self.n_periods:]
        else:
            train,val=outlier_data[:-self.n_periods],outlier_data[-self.n_periods:]
        BE_df=pd.concat([train,val,test])
        return BE_df
            
    
    def main_method_pd(self, all_be_df, us_holidays):
        print("in main_pd")
        final_df = pd.DataFrame()
        for BE in all_be_df[self.target_items].unique().tolist():
            print(BE)
            new_df_monthly=self.holidays_monthly_pd(all_be_df,us_holidays,BE)
            BE_df=self.all_model_combined_pd(new_df_monthly,BE)
#            BE_df[self.target_items]=BE
            final_df = final_df.append(BE_df)
        return final_df

    
    #create config file
    def create_config(self,obj_file):
        with open('data-config.json') as f:
            config_file = json.load(f)
        f.close()
        config_file['raw_data_path'] = self.data_path.replace("\\","/")
        config_file['processed_data_path'] = obj_file
        with open('data-config.json',"w") as f:
            f.write(json.dumps(config_file))
        f.close()
            
    
    #Save files
    def save_transformed_files(self):
        if len(self.transform_data)==0:
            self.transform()
        
        if self.walk_forward == "True":
            save_path ='data/wf_data/'
        else:
            save_path ='data/prod_data/'
            
        data_date = self.summary_df.end_date.max().month_name()+'-'+str(self.summary_df.end_date.max().year)
        if path.exists(save_path+self.overall_be)==False:
            makedirs(save_path+self.overall_be)    
        obj_file = save_path+self.overall_be+'/'+data_date+'.pkl'
        file =  open(obj_file,'wb')
        pickle.dump(self,file)
        file.close()
        self.create_config(obj_file)
        
        
    # Transforms raw Data
    def transform(self):
        
        data_dc = self.combine_data()
        #Create overall billing entity
        overall_be_data = self.create_overall_be(data_dc)
        del data_dc
        #Create granular columns
        granular_data = self.create_granular_col(overall_be_data)
        del overall_be_data
        
        #Remove partial data
        partial_removed_data, summary_df = self.subset_data(granular_data)
        del granular_data
        
        #Update Summary file with billing entity types and check for each of billing entity for minimum of 2 years
        summary_df = self.update_summary(partial_removed_data, summary_df)
        #Recent days missing
        summary_df = self.recent_months_missing(summary_df)
        
        full_data = partial_removed_data.groupby([self.target_items,'MonthYear'])
        full_data = full_data.MonthYear.first().groupby(level=0).size().reset_index()
        full_data.columns = [self.target_items,'count_of_months']
        full_data.sort_values(['count_of_months',self.target_items],ascending=False,inplace=True)
        full_data = full_data.reset_index(drop=True)
    
        summary_df = pd.merge(summary_df,full_data,how='outer',on=self.target_items)
        summary_df.sort_values(['count_of_months',self.target_items],ascending=False,inplace=True)
        
        transformed_data = self.aggregate_data(partial_removed_data)
        summary_df = self.negative_charges_count(transformed_data, summary_df)

        transformed_data=transformed_data.reset_index()
        transformed_data=transformed_data.set_index(self.target_items)    
        transformed_data.drop(list(summary_df[summary_df.count_of_months < 9][self.target_items]), axis=0, inplace=True)    
        transformed_data.drop(list(summary_df[summary_df.missing_recent_months_count > 0][self.target_items]), axis=0, inplace=True)    
        transformed_data=transformed_data.reset_index()
        print(transformed_data.columns)    
        us_holidays = self.usa_holidays()
        
        print("wf_cond",self.walk_forward)
        if self.walk_forward == "True":
            final_df = self.main_method_wf(transformed_data, us_holidays)
        else:
            final_df = self.main_method_pd(transformed_data, us_holidays)
#            final_df.reset_index()
        final_df.to_csv("final_df.csv")

        self.summary_df = summary_df
        self.transform_data = final_df
        del partial_removed_data
        del summary_df