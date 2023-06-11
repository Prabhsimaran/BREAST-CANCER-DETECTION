#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import sklearn.datasets
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)


# In[4]:


# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# print the first 5 rows of the dataframe
data_frame.head()


# In[5]:


# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

# print last 5 rows of the dataframe
data_frame.tail()


# In[6]:


# number of rows and columns in the dataset
data_frame.shape


# In[7]:


# getting some information about the data
data_frame.info()


# In[8]:


# checking for missing values
data_frame.isnull().sum()


# In[9]:


# statistical measures about the data
data_frame.describe()


# In[10]:


# checking the distribution of Target Varibale
data_frame['label'].value_counts()


# In[11]:


data_frame.groupby('label').mean()


# In[12]:


# seperating features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
print(Y)


# In[13]:


#splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)


# In[16]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train, Y_train)


# In[19]:


import sklearn as sk
from sklearn.metrics import confusion_matrix

model_predict = model.predict(X_test)
acc = sk.metrics.accuracy_score(Y_test, model_predict)
print(acc) # Printing accuracy
# Confusion Matrics
conf_matrix = confusion_matrix(Y_test, model_predict)
sns.heatmap(conf_matrix, annot=True, fmt='d');


# In[21]:


from sklearn.linear_model import LogisticRegression

model2 = LogisticRegression()
model2.fit(X_train, Y_train)


# In[22]:


model2_predict = model2.predict(X_test)
acc2 = sk.metrics.accuracy_score(Y_test, model2_predict)
print(acc2)
conf_matrix = confusion_matrix(Y_test, model2_predict)
sns.heatmap(conf_matrix, annot=True, fmt='d');


# In[23]:


#APPLYING K-NEAREST NEIGHBOUR

from sklearn.neighbors import KNeighborsClassifier

model3 = KNeighborsClassifier(n_neighbors=7)
model3.fit(X_train, Y_train)


# In[25]:


model3_predict = model3.predict(X_test)
acc3 = sk.metrics.accuracy_score(Y_test, model3_predict)
print(acc3)
conf_matrix = confusion_matrix(Y_test, model3_predict)
sns.heatmap(conf_matrix, annot=True, fmt='d');


# In[26]:


#APPLYING RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier

model4 = RandomForestClassifier(n_estimators=7)
model4.fit(X_train, Y_train)


# In[27]:


model4_predict = model4.predict(X_test)
acc4 = sk.metrics.accuracy_score(Y_test, model4_predict)
print(acc4)
conf_matrix = confusion_matrix(Y_test, model4_predict)
sns.heatmap(conf_matrix, annot=True, fmt='d');


# In[28]:


from sklearn.ensemble import AdaBoostClassifier

model5 = AdaBoostClassifier(n_estimators=7)
model5.fit(X_train, Y_train)


# In[29]:


model5_predict = model5.predict(X_test)
acc5 = sk.metrics.accuracy_score(Y_test, model5_predict)
print(acc5)
conf_matrix = confusion_matrix(Y_test, model5_predict)
sns.heatmap(conf_matrix, annot=True, fmt='d');


# In[30]:


from sklearn.naive_bayes import GaussianNB
model6 = GaussianNB()
model6.fit(X_train, Y_train)


# In[31]:


model6_predict = model6.predict(X_test)
acc6 = sk.metrics.accuracy_score(Y_test, model6_predict)
print(acc6)
conf_matrix = confusion_matrix(Y_test, model6_predict)
sns.heatmap(conf_matrix, annot=True, fmt='d');


# In[33]:


import matplotlib.pyplot as plt
data = {'SVC':acc, 'LogisticRegression':acc2, 'KNN':acc3, 'RandomForest':acc4, 'AdaBoost':acc5, 'NaiveBayes':acc6}
courses = list(data.keys())
values = list(data.values())
fig = plt.figure()
plt.bar(courses, values, color ='cyan',width = 0.6)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Change in accuracy based on the model")
plt.show()


# In[ ]:




