#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# # Задача 1
# Даны значения величины заработной платы заемщиков банка (zp) и значения их поведенческого кредитного скоринга (ks):
# zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
# ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])
# 
# Найдите ковариацию этих двух величин с помощью элементарных действий, а затем с помощью функции cov из numpy
# Полученные значения должны быть равны.
# Найдите коэффициент корреляции Пирсона с помощью ковариации и среднеквадратичных отклонений двух признаков,
# а затем с использованием функций из библиотек numpy и pandas.

# In[2]:


zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])


# In[5]:


plt.scatter(zp,ks)
plt.xlabel('Величина заработной платы "ZP"')
plt.ylabel('Поведенческий кредитный скоринг "KS"', rotation=90)
plt.show()


# Зависимость линейная. Ожидаем, что между этими данными существует положительная корреляция.
# Вычислим ковариацию двух величин с помощью элементарных действий:

# In[6]:


cov_zp_ks = np.mean(zp*ks) - np.mean(zp)*np.mean(ks)
cov_zp_ks    


# In[7]:


cov_zp_ks = np.cov(zp, ks, ddof=0)[0, 1]
cov_zp_ks


# Вычислим коэффициент корреляции Пирсона с помощью ковариации и среднеквадратичных отклонений двух признаков:
# 
# 

# In[8]:


corr = cov_zp_ks / (np.std(zp) * np.std(ks))
corr


# In[9]:


corr_coef = cov_zp_ks / (np.std(zp, ddof=0) * np.std(ks, ddof=0))
corr_coef


# In[10]:


corr_numpy = np.corrcoef(zp, ks)[0][1]
corr_numpy


# In[11]:


corr_pandas = pd.Series(zp).corr(pd.Series(ks), method='pearson')
corr_pandas


# # Задача 2
# Измерены значения IQ выборки студентов, обучающихся в местных технических вузах:
# 131, 125, 115, 122, 131, 115, 107, 99, 125, 111.
# Известно, что в генеральной совокупности IQ распределен нормально.
# Найдите доверительный интервал для математического ожидания с надежностью 0.95.

# In[12]:


iq = np.array([131, 125, 115, 122, 131, 115, 107, 99, 125, 111])
alpha = 0.05


# Для расчета доверительного интервала при неизвесной СКО будем использовать t-критерий и формулу для среднего арифмитического:

# In[14]:


n = iq.size
std = iq.std(ddof=1)
mean = iq.mean()
print(f' Размер выборки: n = {n}\n'f' Среднее квадратическое отклонение по выборке(несмещенное): {std:.2f}\n'f' Среднее выборочное: {mean:.2f}')


# Значения t и отклонение

# In[24]:


t = stats.t.ppf(1 - alpha / 2, n - 1)
d = t * std / (n) ** 0.5
d,t


# In[26]:


min = mean - d
max = mean + d
print(f'Доверительный интервал для математического ожидания с надежностью 0.95 составляет:{min: .2f};{max: .2f}')


# # Задача 3
# Известно, что рост футболистов в сборной распределен нормально с дисперсией генеральной совокупности, равной 25 кв.см.
# Объем выборки равен 27, среднее выборочное составляет 174.2.
# Найдите доверительный интервал для математического ожидания с надежностью 0.95.
# 
# Для расчета доверительного интервала при извесной СКО будем использовать z-критерий и формулу для среднего арифмитического:

# In[27]:


var = 25
n = 27
mean = 174.2
std = (var)**0.5
alpha = 0.05
z=stats.norm.ppf(1-alpha/2,n-1)
d=z*std/(n)**0.5
min = mean - d
max = mean + d
print(f'Доверительный интервал для математического ожидания с надежностью 0.95 составляет:{min: .2f};{max: .2f}')


# In[ ]:




