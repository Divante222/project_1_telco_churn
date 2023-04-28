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
import numpy as np
import pandas as pd
import math
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np
import wrangle
from scipy import stats
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    '''
    Prepares the telco data to be used
    '''
    # data = data.drop(columns = ['payment_type_id', 'internet_service_type_id',
    #                  'contract_type_id','Unnamed: 0'])
    
    the_columns = data.select_dtypes('object').columns
    
    the_columns = the_columns.drop(['customer_id', 'total_charges','internet_service_type'])
    
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
    

def create_data_dict(df):
    '''
    creates a data dictionary and returns it to be displayed
    '''
    columns = df.columns

    features = []
    definition = ['Customer ID','Whether the customer is a male or a female','Whether the customer is a senior citizen or not','Whether the customer has a partner or not','Whether the customer has dependents or not','Number of months the customer has stayed with the company','Whether the customer has a phone service or not','Whether the customer has multiple lines or not','Whether the customer has online security or not',
                'Whether the customer has online backup or not','Whether the customer has device protection or not','Whether the customer has tech support or not','Whether the customer has streaming TV or not','Whether the customer has streaming movies or not','Whether the customer has paperless billing or not','The amount charged to the customer monthly','The total amount charged to the customer','Whether the customer churned or not',
                'The contract term of the customer (Month-to-month, One year, Two year)','Customer’s internet service provider (DSL, Fiber optic, No)','The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))']

    for i in columns[0:21]:
        features.append(i)
        
    data_dictionary = pd.DataFrame({'features':features,
                                'definition':definition})

    data_dictionary.reset_index(drop=True)
    data_dictionary.index = data_dictionary.features
    data_dictionary = data_dictionary.drop(columns='features')
    left_aligned_df = data_dictionary.style.set_properties(**{'text-align': 'left'})
    left_aligned_df = left_aligned_df.set_table_styles(
    [dict(selector = 'th', props=[('text-align', 'left')])])
     
    return display(left_aligned_df)

def get_barplot_payment_type(train):
    '''
    gets the barplot for payment type
    '''
    plt.title('Does payment type affect churn')
    ax = sns.countplot(x='payment_type', data = train, hue = 'churn_Yes')
    ax.tick_params(axis='x', rotation=30)
    
    plt.show()


def get_barplot_contract_type(train):
    '''
    gets the barplot for contract_type
    '''
    plt.title('Does contract_type type affect churn')
    ax = sns.countplot(x='contract_type', data = train, hue = 'churn_Yes')
    plt.show()

def get_barplot_device_protection(train):
    '''
    gets the barplot for device_protection
    '''
    plt.title('Does device_protection affect churn')  
    ax = sns.countplot(x='device_protection', data = train, hue = 'churn_Yes')
    plt.show()

def get_barplot_tech_support(train):
    '''
    gets the barplot for tech_support
    '''
    
    plt.title('Does tech_support affect churn')
    ax = sns.countplot(x='tech_support_Yes', data = train, hue = 'churn_Yes')
    plt.show()


def chi2_test_function(first, second, name):
    '''
    Runs a chi2 test and prints the result
    '''
    observed = pd.crosstab(first, second)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print('p value: ',p)
    if p < .05:
        print('\nWe reject the null hypothesis because the p value is less than alpha\n')
        print(name ,'and churn_Yes are dependent on one another')
    else:
        print('\nWe fail to reject the null hypothesis because the p value is less than alpha\n')
        print(name ,'and churn_Yes are independent of one another')

def finding_baseline(train):
    '''
    Calculates the baseline
    '''
    train['baseline'] = 0
    baseline = (train.baseline == train.churn_Yes).mean()
    print('Baseline',baseline)
    return baseline

def model_setup(train, validate, test, columns):
    '''
    creates the data to be sent throught the models
    '''
    X_train = train.loc[:, columns]
    X_validate = validate.loc[:, columns]
    X_test = test.loc[:, columns]

    y_train = train.churn_Yes.astype('int')
    y_validate = validate.churn_Yes.astype('int')
    y_test = test.churn_Yes.astype('int')

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def create_csv(tree, X_test, test):
    the_percentages = []
    churn_predict = []

    for i in tree.predict_proba(X_test):
        the_percentages.append(i[0])
        if i[0] > .5:
            churn_predict.append('no')
        else:
            churn_predict.append('yes')
        
        
    the_csv_dataframe = pd.DataFrame({'customer_id':test.customer_id, 
                                    'Probability_of_churn':the_percentages,
                                    'Churn_predict':churn_predict
                                    })
    the_csv_dataframe.to_csv('Churn_Predictions.csv')
    return the_csv_dataframe

def create_descision_tree(X_train,y_train, X_validate, y_validate,max_depth):
    '''
    creating a Decision tree model
    fitting the Descision tree model
    predicting the training and validate data
    '''
    tree = DecisionTreeClassifier(random_state = 123,max_depth=max_depth)
    tree.fit(X_train, y_train)
    train_predict = tree.score(X_train, y_train)
    validate_predict = tree.score(X_validate, y_validate)
    return tree, train_predict, validate_predict, max_depth


def create_random_forest(X_train,y_train, X_validate, y_validate):
    '''
    creating a random_forest model
    fitting the random_forest model
    predicting the training and validate data
    '''
    forest = RandomForestClassifier(random_state = 123)
    forest.fit(X_train, y_train)    
    train_predict = forest.score(X_train, y_train)
    validate_predict = forest.score(X_validate, y_validate)
    return forest, train_predict, validate_predict

def create_logistic_regression(X_train,y_train, X_validate, y_validate,the_c):
    '''
    creating a logistic_regression model
    fitting the logistic_regression model
    predicting the training and validate data
    '''
    logit = LogisticRegression(random_state= 123,C=the_c)
    logit.fit(X_train, y_train)
    train_predict = logit.score(X_train, y_train)
    validate_predict = logit.score(X_validate, y_validate)
    return logit, train_predict, validate_predict


def create_knn(X_train,y_train, X_validate, y_validate):
    '''
    creating a logistic_regression model
    fitting the logistic_regression model
    predicting the training and validate data
    '''
    knn = KNeighborsClassifier(n_neighbors=1,weights='uniform')
    knn.fit(X_train, y_train)
    train_predict = knn.score(X_train, y_train)
    validate_predict = knn.score(X_validate, y_validate)
    return knn, train_predict, validate_predict


def print_statement_for_models(baseline ,train_predict, validate_predict):
    '''
    prints the data from the model
    '''
    print('Baseline',baseline.round(2) * 100)
    print('training data prediciton',train_predict.round(2) * 100)
    print('validate data prediciton',validate_predict.round(2) * 100)

def the_fifth_visual(train, validate, test):
    '''
    creates a random tree classifier model and runs multiple max depths 
    and stores the results in a pandas dataframe
    '''
    columns = validate.columns[21:]
    columns2 = validate.columns[15:17]
    the_list_of_columns = []
    for i in columns:
        if i != 'churn_Yes' and i != 'churn_No':
            the_list_of_columns.append(i)
    for i in columns2:
        the_list_of_columns.append(i)
            

    columns = the_list_of_columns


    train_list = []
    validate_list = []
    features= []
    max_depth_column = []
    X_train, X_validate, X_test, y_train, y_validate, y_test = wrangle.model_setup(train, validate, test, columns)

    for i in range(1,20):        
        tree, train_predict, validate_predict, max_depth= wrangle.create_descision_tree(X_train,y_train, X_validate, y_validate, max_depth=i)
        train_list.append(train_predict)
        validate_list.append(validate_predict)
        features.append(i)
        max_depth_column.append(i)


    the_dataframe = pd.DataFrame({'train':train_list,
                'validate':validate_list,
                'max_depth':max_depth_column
                
                }
                )
    the_dataframe['difference'] = abs(the_dataframe.train - the_dataframe.validate)
    the_dataframe = the_dataframe.sort_values(by='difference')
    the_data = the_dataframe
    the_data = the_data.sort_values(by=['difference','train'], ascending= [True, True])
    the_data =the_data.reset_index()
    the_data = the_data.drop(columns = 'index')

    plt.title('Changing max Depth to find a better model')
    plt.xlabel('index number')
    plt.ylabel('Prediction percentage')
    
    plt.plot(the_data.index, the_data.train, marker='o')
    plt.plot(the_data.index, the_data.validate, marker='o')
    plt.show()
    return the_data

def getting_weights(tree):
    columns = ['tech_support_Yes','device_protection_Yes','contract_type_Two year','contract_type_One year','payment_type_Credit_card','payment_type_Electronic_check','payment_type_Mailed_check']

    the_weight = tree.feature_importances_
    the_weight
    weights_column = []
    for i in the_weight:
        weights_column.append(i)
        
    the_dataframe = pd.DataFrame({'columns': columns, 
                                'the_weight':weights_column})  

    the_dataframe
    plt.title('Weights of initial columns')  

    ax = sns.barplot(x=columns , y=the_weight, data = the_dataframe)
    ax.tick_params(axis='x', rotation=90)
    plt.show()
    

def setup_for_the_extra_model_descision_tree(train,validate, test):
    columns = validate.columns[21:]
    columns2 = validate.columns[15:17]
    the_list_of_columns = []
    for i in columns:
        if i != 'churn_Yes' and i != 'churn_No':
            the_list_of_columns.append(i)
    for i in columns2:
        the_list_of_columns.append(i)
            

    columns = the_list_of_columns

    X_train, X_validate, X_test, y_train, y_validate, y_test = wrangle.model_setup(train, validate, test, columns)
    tree, train_predict, validate_predict, max_depth= wrangle.create_descision_tree(X_train,y_train, X_validate, y_validate, max_depth=6)
    return X_train, X_validate, X_test, y_train, y_validate, y_test, tree


def getting_weights_max(tree, X_train):
    columns = X_train.columns

    the_weight = tree.feature_importances_
    the_weight
    weights_column = []
    for i in the_weight:
        weights_column.append(i)
        
    the_dataframe = pd.DataFrame({'columns': columns, 
                                'the_weight':weights_column})  

    the_dataframe
    plt.title('Weights of all columns')  

    ax = sns.barplot(x=columns , y=the_weight, data = the_dataframe)
    ax.tick_params(axis='x', rotation=90)
    plt.show()

def getting_subgroup(columns, train):
    the_list = []
    for i in columns:
        print(i, 'makes up', (train[i] == 1).mean().round(2) * 100,'percent of the total customer base')
        the_list.append(train[train[i] == 1])

    return the_list

def get_barplot_for_everything(train,name):
    '''
    gets the barplot for everything
    '''
    count = 1
  
    plt.figure(figsize=(20.0, 60.0))
    for i in train.columns[21:]:
        if i != name:
            # plt.title(f'Does {str(i)} affect churn')
            plt.subplot(10, 2, count)
            sns.countplot(x=i, data = train, hue = 'churn_Yes')
            count +=1
            
    
    plt.show()
