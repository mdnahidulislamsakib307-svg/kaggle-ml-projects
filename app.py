#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import streamlit as st


# In[2]:


df = pd.read_csv("C:\\Users\\USER-11\\Downloads\\kc_house_data.csv (1)\\kc_house_data.csv")


# In[6]:


df.shape


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.dtypes


# In[24]:


x = df.drop(['id','date','floors','waterfront','view','sqft_lot','sqft_above','sqft_basement','yr_renovated','sqft_living15','sqft_lot15','price'],axis=1)
y = df['price']


# In[25]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[26]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[27]:


numerical_transformer= Pipeline (steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


# In[28]:


categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])


# In[29]:


preprocessor =ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols ),
    ('cat',categorical_transformer,categorical_cols )
])


# In[30]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[31]:


model =Pipeline(steps=[
    ('pre',preprocessor),('reg',LinearRegression())
])


# In[32]:


model.fit(X_train,y_train)


# In[33]:


y_pred = model.predict(X_test)

print(f'Accuracy:{r2_score(y_pred,y_test)*100:.2f}')


# In[37]:


model2 = Pipeline(steps=[
    ('pre',preprocessor),('reg',RandomForestRegressor(n_estimators=200,random_state=42))
])


# In[40]:


model2.fit(X_train,y_train)


# In[42]:


y_pred2 =model2.predict(X_test)

print(f'Accuracy:{r2_score(y_pred2,y_test)*100:.2f}')


# In[43]:


joblib.dump(model2,'randomforestregressor')


# In[45]:


df.dtypes


# In[ ]:





# In[47]:


load=joblib.load('randomforestregressor')
st.title('House Price Prediction')
bedrooms=st.number_input('bedrooms')
bathrooms=st.number_input('bathrooms')
sqft_living=st.number_input('sqft_living')
grade=st.number_input('grade')
condition=st.number_input('condition')
yr_built=st.number_input('yr_built')
zipcode=st.number_input('zipcode')
lat=st.number_input('lat')
long =st.number_input('long')
if st.button('predict'):
    data = pd.DataFrame({
         'bedrooms':[bedrooms],
        'bathrooms':[bathrooms],
      'sqft_living':[sqft_living],
            'grade':[grade],
        'condition':[condition],
         'yr_built':[yr_built],
          'zipcode':[zipcode],
              'lat':[lat],
             'long':[long]
    })
    prediction =load.predict(data)
    print(f'Price:{prediction[0]}')
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




