#!/usr/bin/env python
# coding: utf-8

# # MULTIPLE LINEAR REGRESSION
# Multiple linear regression'da birden fazla bağımsız(independent) değişkene karşılık bir bağımlı(dependent) değişken bulunur.
# 
# Linear regression veriler arasında var olan korelasyonu(ilişkiyi) kullanarak yeni gelecek verileri tahmin etme modelidir.Burada makine öğrenimi bize veriler arasındaki bu ilişkiyi belirlememize yardımcı olur ve bu sayede yeni verileri tahmin edebiliriz.

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
# sklearn library
from sklearn import linear_model

# veri setlerimizi import ediyoruz, ayraç olarak noktalı virgül olduğu için bunu belirtiyoruz :
df = pd.read_csv("C:\\Users\\pc\\Desktop\\Artificial İntelligence\\PROJELER\\emlak_fiyat_tahmin\\multilinearregression.csv",sep=";")
# veri setimizi görelim ve doğru import ettiğimizi kontrol edelim:
df


# In[11]:


# linear regression modeli tanımlıyoruz :
reg = linear_model.LinearRegression()
reg.fit(df[["alan","odasayisi","binayasi"]],df["fiyat"])
# Predection yapalım,tahmin yapalım
reg.predict([[275,3,11]])


# In[17]:


reg.predict([[230,4,0]]) #230 m2 4 odalı 10 yaşında binanın fiyatını tahmin ediyor


# In[18]:


reg.predict([[355,3,20]]) # 355 m2 3 odalı bina yaşı 20


# In[19]:


reg.predict([[275,4,25]])


# In[20]:


reg.coef_ # a1,a2,a3 katsayılar


# In[21]:


reg.intercept_ # sabit değer


# In[24]:


# Multiple Linear Regression formülümüze dönersek hatırlayalım :
# y = a + b1X1 + b2X2 + b3X3 + ....formülümüzdü
a = reg.intercept_ # sabit değerler
b1 = reg.coef_[0] # katsayılar
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230 # independent değerler(bağımsız değişkeneler)
x2 = 4
x3 = 10
y = a + b1*x1 + b2*x2 + b3*x3

y


# In[ ]:




