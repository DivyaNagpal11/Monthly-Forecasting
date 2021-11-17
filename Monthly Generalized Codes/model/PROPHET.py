
from itertools import product
from tqdm import tqdm
import pandas as pd
import numpy as np
from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')


class PROPHET:

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


    def fit(self,model,train_dataset):
        return model.fit(train_dataset)


    def predict(self,model,data):
        return model.predict(data) 


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


    def optimize_prophet(self,parameters_list,train_dataset,val_dataset):  
        results=[]
        best_adj_mape=float('inf')
        for i in tqdm(parameters_list):
            forecast=pd.DataFrame()
            future=pd.DataFrame()
            
            prophet_basic = Prophet(growth='linear',daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=False,n_changepoints=20,changepoint_prior_scale=i[0])
            prophet_basic.add_seasonality(name='quarterly', period=3, fourier_order=i[1]).add_seasonality(name='biannual', period=6, fourier_order=i[2]).add_seasonality(name='yearly', period=12, fourier_order=i[3])
            prophet_basic=self.fit(prophet_basic,train_dataset)
            
            future= prophet_basic.make_future_dataframe(periods=int(self.n_periods))
            forecast=self.predict(prophet_basic,future)
            
            y_true=np.array(list(train_dataset['y']))
            y_pred=np.array(list(forecast.yhat[:-int(self.n_periods)]))
            val_predicted=np.array(list(forecast.yhat[-int(self.n_periods):]))
            train_mape=round((self.mean_absolute_percentage_error(y_true,y_pred)),2)
            val_mape=round((self.mean_absolute_percentage_error(val_dataset["y"],val_predicted)),2)
            adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_dataset))+val_mape*len(val_dataset)/(len(y_true)+len(val_dataset))
            
            if adj_mape <= best_adj_mape:
                best_adj_mape=adj_mape
                best_model = prophet_basic
                
            results.append([i,train_mape,val_mape,adj_mape])

        result_table=pd.DataFrame(results,columns=['parameters','train_mape','val_mape','adj_mape'])
        result_table=result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
        return result_table, best_model 
    
    
    def run_model(self):
        print("In prophet")
        self.train=self.train.reset_index()
        train_dataset= pd.DataFrame()
        val_dataset= pd.DataFrame()
        val_set=self.train[-(int(self.n_periods)):]
        train_dataset['ds'] = self.train[self.target_dates]
        train_dataset['y']=self.train[self.target_value]
        val_dataset['ds'] = val_set[self.target_dates]
        val_dataset['y']=val_set[self.target_value]
    
        result_table, best_model = self.optimize_prophet(self.parameters_list,train_dataset,val_dataset)
        future= best_model.make_future_dataframe(periods=int(self.n_periods))
        
        overall_train=pd.concat([self.train,val_set])
        overall_train['ds'] = overall_train[self.target_dates]
        overall_train['y'] = overall_train[self.target_value]
        fitted_val_list=[]
        
        c=1
        for cp,fo3,fo6,fo12 in result_table.parameters:
            try:
                if c > 5:
                    break
                prophet_basic1 = Prophet(growth='linear',daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=False,n_changepoints=20,changepoint_prior_scale=cp)
                prophet_basic1.add_seasonality(name='quarterly', period=int(self.n_periods), fourier_order=fo3).add_seasonality(name='biannual', period=6, fourier_order=fo6).add_seasonality(name='yearly', period=12, fourier_order=fo12)
                prophet_basic1.fit(overall_train)
                future= prophet_basic1.make_future_dataframe(periods=int(self.n_periods))
                forecast=self.predict(prophet_basic1,future).yhat[-int(self.n_periods):]
                c=c+1
                get_list=[]
                for i in range(len(forecast)):
                    get_list.append(forecast.iloc[i])
                fitted_val_list.append(get_list)
            except:
                continue
            
        fitted_val=pd.DataFrame(fitted_val_list,columns=[x for x in range(1,self.n_periods+1)])
        fitted_mean=[]
        for i in range(1,self.n_periods+1):
            fitted_mean.append(fitted_val[i].mean())
        
#        print("in prophet fitted mean",fitted_mean)
        if self.walk_forward == "True":
            test_set1=np.array(list(self.test[self.target_value]))
            test_mape=round(self.mean_absolute_percentage_error(test_set1,fitted_mean),2)
            one_month=round(self.mean_absolute_percentage_error(test_set1[0],fitted_mean[0]),2)
            two_month=round(self.mean_absolute_percentage_error(test_set1[1],fitted_mean[1]),2)
            third_month=round(self.mean_absolute_percentage_error(test_set1[2],fitted_mean[2]),2)
            self.results.append([self.target_item,"PROPHET",self.set_no,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_mape,test_set1,fitted_mean,one_month,two_month,third_month])
            self.mape_res=pd.DataFrame(self.results,columns=[self.target_items,'model','set_no','train_mape','val_mape','test_mape','test_actual','test_predicted','one_month','two_month','third_month'])
        else:
            self.results.append([self.target_item,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),list(self.test[self.target_dates]),fitted_mean])
            self.mape_res=pd.DataFrame(self.results,columns=[self.target_items,'train_mape','val_mape','test_dates','test_predicted'])