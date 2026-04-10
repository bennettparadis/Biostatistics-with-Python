# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:30:43 2026

@author: bparadis
"""
#compare multiple groups HbA1c based on BMI level
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#load the data
data=pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Dataset of Diabetes.csv")

#filter data
filtered_data = data[data['CLASS'].isin(['Y'])].copy()

#create new column for weight class based on BMI
def weight_class(bmi): 
    if bmi < 25:
        return 'normal'
    elif bmi >= 25 and bmi < 30:
        return 'overweight'
    else:
        return 'obese'
    
#apply BMI function
filtered_data['weight_class'] = filtered_data['BMI'].apply(weight_class)  

#create a table for descriptive stats of each class
normal_stats = filtered_data[
    filtered_data['weight_class'] == 'normal']['HbA1c'].describe()
                                              
overweight_stats = filtered_data[
    filtered_data['weight_class'] == 'overweight']['HbA1c'].describe()                                           
                                              
obese_stats = filtered_data[
    filtered_data['weight_class'] == 'obese']['HbA1c'].describe()

stats_table = pd.DataFrame({'Normal':normal_stats, 'Overweight':overweight_stats, 'Obese':obese_stats})
print(stats_table)
 
#perform ANOVA
fvalue, pvalue = stats.f_oneway(
    filtered_data[filtered_data['weight_class'] == 'normal']['HbA1c'], 
    filtered_data[filtered_data['weight_class'] == 'overweight']['HbA1c'],
    filtered_data[filtered_data['weight_class'] == 'obese']['HbA1c']
        )
    
results = pd.DataFrame({'F-value': [fvalue],
                        'p-value': [pvalue]})

print(f"ANOVA Results:\n {results}")

#Tukey-Kramer post hoc test tells us more about group to group comparisons
from statsmodels.sandbox.stats.multicomp import MultiComparison
mc = MultiComparison(
    filtered_data['HbA1c'], filtered_data['weight_class'])
result = mc.tukeyhsd()
print('Tukey-Kramer Post Hoc Test:')
print(result)














