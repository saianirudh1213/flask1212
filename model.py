

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import pickle

# In[2]:


dataset = pd.read_csv('data.csv')



# # Step 3: Data Preprocessing

# In[13]:


dataset_X = dataset.iloc[:,0:4].values
dataset_Y = dataset.iloc[:,-1].values


# In[14]:


dataset_X


# In[15]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


# In[16]:


dataset_scaled = pd.DataFrame(dataset_scaled)


# In[17]:


X = dataset_scaled
Y = dataset_Y


# In[18]:


X


# In[19]:


Y


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset['Infected with Covid19'] )


# # Step 4: Data Modelling

# In[25]:


# Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 42)
logreg.fit(X_train, Y_train)


logreg.score(X_test, Y_test)


# In[27]:


Y_pred = logreg.predict(X_test)

# here we using the knn because it have a highest accuracy 78%



pickle.dump(logreg, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict(sc.transform(np.array([[16, 10, 13, 15]]))))


