#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np 
import pandas as pd 


# In[33]:


marks = pd.read_csv("C:/Users/Asifa Zehra/Downloads/archive/Student_Marks.csv")
print(marks.shape, '\n')
marks.head(10)


# In[34]:


marks.isnull().mean()


# In[35]:


import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize': [12.0, 8.0],
                     'font.size': 14})


# In[46]:


plt.scatter(x=marks['number_courses'], y=marks['Marks'], c='g')
plt.xlabel("Number of Courses")
plt.ylabel("Marks")
plt.title("Marks vs Number of Courses  ");


# In[47]:


plt.scatter(x=marks['time_study'], y=marks['Marks'], c='r')
plt.xlabel("Study Timing")
plt.ylabel("Marks")
plt.title("Marks vs Study Timing  ");


# In[48]:


plt.scatter(x=marks['time_study'], y=marks['number_courses'], c='c')
plt.xlabel("Study Timing")
plt.ylabel("Number of Courses")
plt.title("Study Timing vs Number of Courses");


# In[39]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = marks[['number_courses', 'time_study']]
y = marks['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

sc = MinMaxScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[40]:


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)


# In[41]:


from sklearn.metrics import mean_absolute_error


# In[42]:


print(f"Linear Regression: MAE: {mean_absolute_error(y_test, lr_preds)}")


# In[43]:



student_1 = {'num_courses': 7, 'study_time': 7}
student_2 = {'num_courses': 2, 'study_time': 5}


# In[44]:


print(f"Marks of Student 1 will be: {rf_model.predict(np.array([list(student_1.values())]))[0]}")
print(f"Marks of Student 2 will be: {rf_model.predict(np.array([list(student_2.values())]))[0]}")


# In[ ]:




