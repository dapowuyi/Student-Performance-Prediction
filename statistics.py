#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#funtion to calculate count
def count(df):
    countList = df.select_dtypes(include=["int64", "float64", "datetime64"]).count()
    return countList

#funtion to calculate mean
def mean(df):
    mean = np.mean(df.select_dtypes(include=["int64", "float64", "datetime64"]))
    return mean

#funtion to calculate maximum value
def maxValue(df):
    maxValues = np.max(df.select_dtypes(include=["int64", "float64", "datetime64"]))
    return maxValues

#funtion to calculate minimum value
def minValue(df):
    minValues = np.min(df.select_dtypes(include=["int64", "float64", "datetime64"]))
    return minValues

#funtion to calculate standard deviation
def standardDev(df):
    stdev = round(np.std(df.select_dtypes(include=["int64", "float64", "datetime64"])), 2)
    return stdev

#funtion to calculate variance
def variance(df):
    vrn = round(np.var(df.select_dtypes(include=["int64", "float64", "datetime64"])),2)
    return vrn

#funtion to calculate twenty five percentile
def twentyFivePercentile(df):
    twentyFive = np.percentile(df.select_dtypes(include=["int64", "float64", "datetime64"]), 25)
    return twentyFive

#funtion to calculate fifty five percentile
def fiftyFivePercentile(df):
    fifty = np.percentile(df.select_dtypes(include=["int64", "float64", "datetime64"]), 50)
    return fifty

#funtion to calculate seventy five percentile
def seventyFivePercentile(df):
    seventyFive = np.percentile(df.select_dtypes(include=["int64", "float64", "datetime64"]), 75)
    return seventyFive

#funtion to calculate skewness
def skewness(df):
    skw = pd.DataFrame(df.select_dtypes(include=["int64", "float64", "datetime64"])).skew()
    return skw

def kurt(df):
    kut = pd.DataFrame(df.select_dtypes(include=["int64", "float64", "datetime64"])).kurtosis()
    return kut


#function for feature importance plot

def featureImportancePlot(importance, columnNames, model):

#Create arrays from the feature importance and the feature names
    featureImportance = np.array(importance)
    featureNames = np.array(columnNames)
#Create a DataFrame using a Dictionary
    data = {'featureNames':featureNames,'featureImportance':featureImportance}
    new_df = pd.DataFrame(data)
#Sort the DataFrame in order decreasing feature importance
    new_df.sort_values(by=['featureImportance'], ascending=False,inplace=True)
    
#Defining the plot size
    plt.figure(figsize=(8,6))
#Ploting the Searborn bar chart
    sns.barplot(x=new_df['featureImportance'], y=new_df['featureNames'])
#Add chart labels
    plt.title(model + ' FEATURE IMPORTANCE')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    

