import pandas as pd 
import env
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from sklearn.impute import SimpleImputer

def split_data(df, target):
    '''
    Takes in the titanic dataframe and return train, validate, test subset dataframes
    '''
    
    
    train, test = train_test_split(df,
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df[target]
                                   )
    train, validate = train_test_split(train, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train[target]
                                       )
    
    return train, validate, test



def prep_telco(data):
    # data = data.drop(columns = ['payment_type_id', 'internet_service_type_id',
    #                  'contract_type_id','Unnamed: 0'])
    
    the_columns = data.select_dtypes('object').columns

    the_columns = the_columns.drop(['customer_id', 'total_charges'])

    dummy = pd.get_dummies(data[the_columns], drop_first=True)

    # for i in dummy.columns:
    #     dummy[i] = dummy[i].str.replace('0', 0)
    #     dummy[i] = dummy[i].str.replace('1', 1)

    data = pd.concat([data, dummy], axis=1)


    data = data.T.drop_duplicates().T
    data = data.drop(columns = ['payment_type_id','internet_service_type_id','contract_type_id'])
    data.total_charges = data.total_charges.str.replace(' ', '0')
    data.total_charges = data.total_charges.astype(float) 
    data = data.rename(columns={"payment_type_Credit card (automatic)": "payment_type_Credit_card",
                                "payment_type_Electronic check":'payment_type_Electronic_check',
                                "payment_type_Mailed check":'payment_type_Mailed_check'
                                })
    the_columns = data.columns

    for i in data.columns[21:]:
        data[i] = data[i].astype(int)
    
    return data 


def new_telco_data(SQL_query, url):
    '''
    this function will:
    - take in a SQL_query 
    -create a connection url to mySQL
    -return a df of the given query from the telco_churn
    
    '''
    url= f'mysql+pymysql://{env.username}:{env.password}@{env.hostname}/telco_churn'
    return pd.read_sql(SQL_query,url)    
        

    
def get_telco_data(filename = "telco_churn.csv"):
    '''
    this function will:
    -check local directory for csv file
        return if exists
    if csv doesn't exist
    if csv doesnt exist:
        - create a df of the SQL_query
        write df to csv
    output telco_churn df
    
    '''
    SQL_query = '''
    select *
    from customers
    join contract_types using(contract_type_id)
    join internet_service_types using(internet_service_type_id)
    join payment_types using(payment_type_id)
    ;
    '''    
    directory = os.getcwd()
    filename = 'telco_churn.csv'
    
    url= f'mysql+pymysql://{env.username}:{env.password}@{env.hostname}/telco_churn'
    
    if os.path.exists(directory + filename):
        df = pd.read_csv(filename)
        return df
    else:
        df = new_telco_data(SQL_query, url)
        df.to_csv(filename)
        return df