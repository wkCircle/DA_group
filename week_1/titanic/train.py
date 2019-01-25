import pandas as pd
import feature
from sklearn.linear_model import LogisticRegression #Logistic regression
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
import pdb

'''
Train data set and test data set
'''
train_set,test_set = train_test_split(feature.train_data,test_size=0.3)
train_label = feature.label_data.loc[train_set.index]
test_label = feature.label_data.loc[test_set.index]
#Training for validation set
model = LogisticRegression()
model.fit(train_set,train_label)
prediction = model.predict(test_set)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_label))
#Train whole dataset
use_model = LogisticRegression()
use_model.fit(feature.train_data,feature.label_data) 
result = pd.DataFrame(use_model.predict(feature.test_data),columns=['Survived'])
result['PassengerId'] = pd.read_csv('test.csv')['PassengerId']
result.to_csv('result.csv',index=None)
#TBA:
'''
1.調參(GridSearchCV)
2.Ensemble方法
'''
