#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df= pd.read_csv('C:/Users/611517676/Pictures/Dataset/processed_acl/dvd/7282_1.csv')


# In[4]:


df.head()


# In[5]:


df=df[df['reviews.rating']!=3.0]
df['Positive Rating']=np.where(df['reviews.rating']>3.0,1.0,0.0)
df.head()


# In[19]:


df.info()


# In[20]:


df['reviews.text'].fillna('ND',inplace=True)


# In[21]:


df.info()


# In[22]:


df['Positive Rating'].mean()


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df['reviews.text'],df['Positive Rating'],random_state=5)


# In[24]:


print('X_Train first entry',x_train[0])
print('X_Train Shape', x_train.shape)


# In[25]:


from sklearn.feature_extraction.text import CountVectorizer


# In[26]:


vect=CountVectorizer().fit(x_train)


# In[27]:


vect.get_feature_names()[::2000]


# In[28]:


len(vect.get_feature_names())


# In[29]:


x_train_vectorized=vect.transform(x_train)
x_train_vectorized


# In[30]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train_vectorized,y_train)


# In[31]:


from sklearn.metrics import roc_auc_score
predictions=model.predict(vect.transform(x_test))
print('AUC:',roc_auc_score(y_test,predictions))


# In[32]:


feature_name=np.array(vect.get_feature_names())
feature_name


# In[35]:


sorted_coef_index=model.coef_[0].argsort()


# In[36]:


print('Smallest Coef:\n{}\n'.format(feature_name[sorted_coef_index[:10]]))

print('Largest Coef:\n{}\n'.format(feature_name[sorted_coef_index[:-11:-1]]))


# # TF-IDF

# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect1 =TfidfVectorizer(min_df=5).fit(x_train)
len(vect1.get_feature_names())


# In[38]:


x_train_vectorized=vect.transform(x_train)


# In[40]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train_vectorized,y_train)
predictions=model.predict(vect.transform(x_test))
print('AUC:',roc_auc_score(y_test,predictions))


# In[41]:


feature_name=np.array(vect.get_feature_names())
sorted_tfidf_index=x_train_vectorized.max(0).toarray()[0].argsort()
print('Smallest tfidf:\n{}\n'.format(feature_name[sorted_tfidf_index[:10]]))
print('Largest tfidf:\n{}\n'.format(feature_name[sorted_tfidf_index[:-11:-1]]))


# In[42]:


feature_name=np.array(vect.get_feature_names())
sorted_coef_index=model.coef_[0].argsort()
print('Smallest Coef:\n{}\n'.format(feature_name[sorted_coef_index[:10]]))
print('Largest Coef:\n{}\n'.format(feature_name[sorted_coef_index[:-11:-1]]))


# In[43]:


print(model.predict(vect.transform(['not an issue,phone is working','an issue,phone is not working'])))


# In[44]:


vect =CountVectorizer(min_df=5,ngram_range=(1,2)).fit(x_train)
x_train_vectorized=vect.transform(x_train)


# In[45]:


len(vect.get_feature_names())


# In[47]:


model=LogisticRegression()
model.fit(x_train_vectorized,y_train)
predictions=model.predict(vect.transform(x_test))
print('AUC:',roc_auc_score(y_test,predictions))


# In[48]:


feature_name=np.array(vect.get_feature_names())
sorted_coef_index=model.coef_[0].argsort()
print('Smallest Coef:\n{}\n'.format(feature_name[sorted_coef_index[:10]]))
print('Largest Coef:\n{}\n'.format(feature_name[sorted_coef_index[:-11:-1]]))


# In[49]:


print(model.predict(vect.transform(['not an issue,phone is working','an issue,phone is not working'])))


# In[ ]:




