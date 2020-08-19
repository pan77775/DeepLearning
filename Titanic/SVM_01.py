#Common Model Algorithms
import numpy as np
import pandas as pd

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import model_selection

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

#Import data from file: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
data_org = pd.read_csv("input/train.csv")
data_test_org = pd.read_csv("input/test.csv")

#Data info
print(data_org.head())
print(data_org.info())

#Create a copy data to use
data_train = data_org.copy(deep = True)
data_test = data_test_org.copy(deep = True)

#Make a list to clean both datasets at once
data_cleaner = [data_train, data_test]

#Clean data
for dataset in data_cleaner:
    #Missing value in Fare and Embarked (Age later)
    dataset.describe(include = 'all')
    #Complete Embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    #Complete missing Fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
    #New feature Title from name
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
    title_names = (dataset['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    
    #New feature Family size
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    
    #New feature IsAlone
    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0 # now update to no/0 if family size is greater than 1
       
    #Create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
    dummy_pclass = pd.get_dummies(dataset['Pclass'])
    dummy_pclass.columns = ['Class_1','Class_2','Class_3']
    dummy_pclass.drop(['Class_3'], axis=1, inplace=True)
    dataset.drop(['Pclass'], axis=1, inplace=True)
    dataset['Class_1'] = dummy_pclass['Class_1']
    dataset['Class_2'] = dummy_pclass['Class_2']
    
    #Drop some feature
    drop_feature=['PassengerId','Name','Ticket','Cabin']
    dataset.drop(drop_feature, axis=1, inplace=True)
   
    
#Define x and y variables for dummy features original
train_dummy = pd.get_dummies(data_train)
test_dummy = pd.get_dummies(data_test)

#Complete missing age with RF
#Split data to train_age and missing_age
train_data_age = train_dummy['Age']>0
train_age_x = train_dummy.drop(['Age','Survived'], axis=1, inplace=False).loc[train_data_age]
train_age_y = train_dummy['Age'].loc[train_data_age]
train_missing_age_x = train_dummy.drop(['Age','Survived'], axis=1, inplace=False).loc[train_data_age == False]
test_data_age = test_dummy['Age']>0
test_missing_age_x = test_dummy.drop(['Age'], axis=1, inplace=False).loc[test_data_age == False]
#Fitting RandomForest to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
regressor.fit(train_age_x, train_age_y)
train_dummy.loc[train_data_age == False, 'Age'] = regressor.predict(train_missing_age_x)
test_dummy.loc[test_data_age == False, 'Age'] = regressor.predict(test_missing_age_x)

#If age under is 12 ,set sex to child(not male or female) 
train_dummy.loc[train_dummy['Age'] <= 12, 'Sex_male'] = 0
train_dummy.loc[train_dummy['Age'] <= 12, 'Sex_female'] = 0
test_dummy.loc[test_dummy['Age'] <= 12, 'Sex_male'] = 0
test_dummy.loc[test_dummy['Age'] <= 12, 'Sex_female'] = 0

#Split data to train and cv
train_x, cv_x, train_y, cv_y = model_selection.train_test_split(train_dummy.drop(['Survived'], axis=1, inplace=False), train_dummy['Survived'], test_size = 0.2, random_state = None)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_x = sc_X.fit_transform(train_x)
cv_x = sc_X.transform(cv_x)

#Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(C=1, kernel = 'rbf' , random_state = 0)
classifier.fit(train_x, train_y)

#Predict cv_y
y_pred = classifier.predict(cv_x)

#Accuracy about CV
from sklearn.metrics import accuracy_score
acc = accuracy_score(cv_y, y_pred, normalize=True, sample_weight=None)

#Test_x Feature scaling
test_x = sc_X.fit_transform(test_dummy)

#Predict test data
y_pred_test = classifier.predict(test_x)

#Output 
ans = pd.DataFrame(data_test_org['PassengerId'])
ans['Survived'] = y_pred_test
ans.to_csv('output/99_C1.csv', index = False, header = True)
