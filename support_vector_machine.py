#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
class SVM:
    
    def __int__(self,lr=0.001,lambda_param=0.01,n=1000):
        self.lr=lr
        self.lambda_param=lambda_param
        self.n=n
        self.w=None
        self.b=None
        
        
    def fit(self,X,y):
        #init paramters
        y_=np.where(y<=0,-1,1)
        n_samples,n_features=X.shape
        self.w=np.zeros(n_features)
        self.b=0
        #gradient decent
        for _ in range(self.n):
            for idx,x_i in enumerate(X):
                condition=y_[idx]* (np.dot(x_i,self.w)-self.b)>=1
                if condition:
                    self.w-=self.lr*2*self.lambda_param*self.w
                    
                else:
                    self.w-=self.lr*(2*self.lambda_param*self.w  -  np.dot(x_i,y_[idx]))
                    
                    self.b-=self.lr*(y_[idx])
                    
    
    
    def prdict(self,X):
        
        linear_output=np.dot(X,self.w)-self.b
        return np.sign(linear_output)
        
    


# In[ ]:




