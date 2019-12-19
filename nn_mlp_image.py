
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
neural_model= MLPClassifier()


# In[2]:


samp=7
train_data=np.zeros((7*40,10304))
train_label=np.zeros((7*40))
count=-1
plt.figure(1)
plt.ion()
for i in range(1,41):
    for j in range(1,samp+1):
        #plt.cla()
    
        count=count+1
        #print(count)
        path ='C:\\Users\\KAVITA.LAPTOP-E2HD8FAQ\\Contacts\\Desktop\\orl_face\\u%d\\%d.png'%(i,j)
        im=mimg.imread(path)
        feat=im.reshape(1,-1)
        train_data[count,:]=feat
        train_label[count]=i
test_data=np.zeros(((10-samp)*40,10304))
test_label=np.zeros((10-samp)*40)
count=-1
for i in range(1,41):
    for j in range(samp+1,11):
        #plt.cla()
        count=count+1
        path ='C:\\Users\\KAVITA.LAPTOP-E2HD8FAQ\\Contacts\\Desktop\\orl_face\\u%d\\%d.png'%(i,j)
        im=mimg.imread(path)
        feat=im.reshape(1,-1)
        test_data[count,:]=feat
        test_label[count]=i


# In[3]:


classifier=MLPClassifier(activation='relu',hidden_layer_sizes=(100,50), learning_rate='constant', max_iter=200,solver='adam',verbose=True)
classifier.fit(train_data,train_label)
y_pred=classifier.predict(test_data)
c = confusion_matrix(test_label, y_pred)
print(c)
a1=classifier.score(test_data,test_label)
print(a1)


# In[4]:


a=classifier.score(test_data,test_label)


# In[5]:


a

