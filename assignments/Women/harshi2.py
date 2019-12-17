#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('women.csv')
data.head(6)


# In[4]:


data.shape


# In[5]:


data.info()


# In[10]:


sns.heatmap(data.corr(),cmap='coolwarm',xticklabels=True,annot=True)
plt.title('data.corr()')


# In[11]:


data['Review Text'].unique()


# In[12]:


data.drop('Title', axis = 1,inplace=True)


# In[13]:


data.info()


# In[14]:


data['Division Name'].unique()


# In[15]:


data['Division Name'].value_counts()


# In[17]:


diviname= pd.get_dummies(data['Division Name'])
diviname.head()


# In[18]:


data['Department Name'].unique()


# In[19]:


data['Department Name'].value_counts()


# In[20]:


deptname= pd.get_dummies(data['Department Name'])
deptname.head()


# In[21]:


data['Class Name'].unique()


# In[22]:


classname= pd.get_dummies(data['Class Name'])
classname.head()


# In[23]:


data = pd.concat([data,diviname,classname,deptname],axis=1)


# In[24]:


data.info()


# In[25]:


old_data = data.copy()
data.drop(['Division Name','Department Name','Class Name'],axis=1,inplace=True)
data.head()


# In[26]:


data.drop('Review Text', axis = 1,inplace=True)


# In[27]:


data.info()


# In[28]:


data.drop('Unnamed: 0', axis = 1,inplace=True)


# In[29]:


data.info()


# In[30]:


data['answer'] = pd.Series(np.random.randn(len(data['Tops'])), index=data.index)


# In[31]:


data.info()


# In[32]:


data.head()


# In[35]:


def impute_answer(cols):
    Answer= cols[0]
    Rating= cols[1]
    if Rating >= 3:
            return 1
    else:
            return 0    
        
    
        

    


# In[37]:


data['answer'] = data[['answer','Rating']].apply(impute_answer,axis=1)


# In[39]:


data.head()


# In[42]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('answer',axis=1),data['answer'], test_size=0.30,random_state=101)
                                                     
                                                    


# In[43]:


from sklearn.linear_model import LogisticRegression

# Build the Model.
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[44]:


predicted =  logmodel.predict(X_test)
predicted


# In[45]:


from sklearn.metrics import precision_score

print(precision_score(y_test,predicted))


# In[46]:


from sklearn.metrics import f1_score

print(f1_score(y_test,predicted))


# In[47]:


from sklearn.metrics import classification_report

print(classification_report(y_test,predicted))


# In[ ]:




