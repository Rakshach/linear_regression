#!/usr/bin/env python
# coding: utf-8

#  ### Written by :Raksha Choudhary<br>
#  ### Data Science and Bussiness Analytics Internship<br>
#  ### GRIP The Sparks Foundation<br>
#  ### Task 1: Predic the percentage of marks that student expected to scores based upon the number of hours they spend

# ####  In this we use simple linear regression involving two variables

# ### import libraries

# In[3]:


#importing all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### load Dataset

# In[5]:


#take data from link
data="http://bit.ly/w-data"
s_data=pd.read_csv(data)

print("Data imported successfully")


# In[6]:


s_data.head(10)


# ### Plot the graph for better visualisation

# In[54]:


#plot the distribution score
s_data.plot(x="Hours",y="Scores",style="o",c='y')
plt.title('Hours Vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.style.use("dark_background")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# #### From the graph above we conclude that there is positive linear relation between "number of hours studied " and " percentage of score"

# ### Preparing the data

# In[11]:


#prepare the data 

x= s_data.iloc[:,:-1].values
y= s_data.iloc[:,1].values


# #### In above we divide the data into attributes and labels.After that we split the data into test and train data by using split() method which is inbuilt in scikit learn

# In[12]:


#split into train and test data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# #### random_state ensures that random numbers are generated in the same order.However if the person use a particular value (1 or 0)everytime the result will be same i.e same values in train and test datasets.

# ### Training the algorithm

# In[16]:


#train the algorithm

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

print("Training Done!!!!")


# In[17]:


#plot regression line
line = reg.coef_*x+reg.intercept_


# In[49]:


#plot the test data
plt.scatter(x,y,c='g')
plt.plot(x,line,c='y')
plt.style.use("dark_background")
plt.tight_layout()
plt.show()


# ### Making predictions

# In[19]:


#making prediction

print(x_test)
y_pred = reg.predict(x_test)


# In[20]:


#Compare Actual Vs Predicted

df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# In[22]:


# test your own data
hours = np.array([8.63,4,6.0,9.81])
own_pred = reg.predict(hours.reshape(-1,1))
print("No. of hours={}".format(hours))
print("Predicted Scores={}".format(own_pred[0]))
print("Predicted Scores={}".format(own_pred[1]))
print("Predicted Scores={}".format(own_pred[2]))
print("Predicted Scores={}".format(own_pred[3]))


# ### Evaluting the model

# In[23]:


#Evalute the model

from sklearn import metrics
print("Mean absolute Error:",metrics.mean_absolute_error(y_test,y_pred))


# In[24]:


from sklearn.metrics import r2_score
print("R_squard:",metrics,metrics.r2_score(y_test,y_pred))


# In[25]:


from sklearn.metrics import mean_squared_error
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))


# #### This step is important to compare how well different model perform on a particular dataset.For simplicity we uses mean square error.

# 
