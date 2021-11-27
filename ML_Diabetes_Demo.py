#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.linear_model import LogisticRegression
from  xgboost import XGBClassifier
from azureml.core import Workspace, Experiment, ScriptRunConfig

# get workspace
# ws = Workspace.from_config()

# # get compute target
# target = ws.compute_targets['target-name']

# # get registered environment
# env = ws.environments['env-name']

# # get/create experiment
# exp = Experiment(ws, 'experiment_name')

# # set up script run configuration
# config = ScriptRunConfig(
#     source_directory='.',
#     script='script.py',
#     compute_target=target,
#     environment=env,
#     arguments=['--meaning', 42],
# )

# # submit script to AML
# run = exp.submit(config)
# print(run.get_portal_url()) # link to ml.azure.com
# run.wait_for_completion(show_output=True)


# In[5]:


df=pd.read_csv('./diabetes.csv')


# In[6]:


df.head()


# In[7]:


df.duplicated().sum()


# In[8]:


df.isnull().sum()


# In[9]:


df.isna().sum()


# In[10]:


df.shape


# In[11]:


df.describe()


# In[12]:


df.info()


# In[13]:


df.skew()


# In[20]:


pd.DataFrame.hist(df,figsize=(10,15))
plt.show()


# In[22]:


pd.DataFrame.boxplot(df,figsize=(15,10))


# In[25]:


print(df['Age'].max(),df['Age'].min())


# In[26]:


df.Insulin.head()


# In[28]:


df.head()


# In[38]:


df['Insulin'].describe()


# In[42]:


df[df['Insulin'] > 250].count()


# In[43]:


df['SkinThickness'].plot(kind='kde')


# In[49]:


df['SkinThickness']


# In[54]:


df[df['SkinThickness'] ==0].count()


# In[57]:


opt1=np.log(df['SkinThickness']-(min(df['SkinThickness'])-1))
pd.Series(opt1).plot(kind='kde')


# In[61]:


cor=df.corr()
sns.heatmap(cor,cmap='Blues')
plt.figure(figsize=(10,8))
plt.show()


# In[65]:


for i in df.columns:
    plt.figure(figsize=(4,5))
    sns.kdeplot(x=i,hue='Outcome',data=df)
    plt.show()


# In[95]:


df1=df.copy(deep=True)


# In[96]:


y=df[['Outcome']]
X=df.drop('Outcome',axis=1)


# In[107]:


scale=MinMaxScaler()
X=scale.fit_transform(X)


# In[108]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)


# In[133]:


base_models = {'Logistic Regression': LogisticRegression(max_iter=250),
 'KNN':  KNeighborsClassifier(n_neighbors=30, weights="distance"),
'Random_forest' : RandomForestClassifier(n_estimators=300,criterion='gini',max_depth=20,random_state=0),
'XGB':XGBClassifier(n_estimators=250,max_depth=5,base_score=0.75)}


# In[ ]:


for name in base_models:
    model=(base_models[name].fit(X_train,y_train))
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)

    cm1 = confusion_matrix(y_test, y_pred_test)
    print(cm1)
    acc=accuracy_score(y_test,y_pred_test)
    print(f'The accuracy of {name} is {acc}')


# In[139]:


logis=LogisticRegression(max_iter=250)
l1=logis.fit(X_train,y_train)
y_pred=l1.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[ ]:





# In[ ]:




