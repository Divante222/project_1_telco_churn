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



dict_for_dataframe = {}
df = wrangle.get_telco_data()
df = wrangle.prep_telco(df)

train, validate, test = wrangle.split_data(df, 'churn_Yes')
columns = train.columns[21:32]
the_list_of_columns = []
for i in columns:
    if i != 'churn_Yes':
        the_list_of_columns.append(i)
columns = the_list_of_columns

the_c_list = [.01, .1, 1, 10, 100, 1000]
train_list = []
validate_list = []
features= []

X_train, X_validate, X_test, y_train, y_validate, y_test = wrangle.model_setup(train, validate, test, columns)

for num_cols in range(2, len(columns)+1):
    for i in itertools.combinations(columns, num_cols):
        print(i)
        train_score, validate_score, test_score = wrangle.create_descision_tree(X_train,y_train, X_validate, y_validate)
        train_list.append(train_score)
        validate_list.append(validate_score)
        features.append(i)
        
the_dataframe = pd.DataFrame({'train':train_list,
             'validate':validate_list,
             'features':features
             }
             )
the_dataframe['difference'] = abs(the_dataframe.train - the_dataframe.validate)
the_dataframe = the_dataframe.sort_values(by='difference')
print(the_dataframe)