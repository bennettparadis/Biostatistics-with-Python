# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:53:40 2026

@author: bparadis
"""

#multiple linear regression example
#association between urea and HbA1c in diabetes subjects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

#load,  clean, & filter the data for patients with T2DM
data=pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Dataset of Diabetes.csv")
data = data.replace('f', 'F', regex = True)
filtered_data = data[data['CLASS'].isin(['Y'])].copy()

#Model 1 
model1 = smf.ols(
    formula = 'HbA1c ~ Urea + Gender + AGE + HDL', data = data).fit()
sns.regplot(x='Urea', y = 'HbA1c', data =data, ci = 95)
plt.show()

#nearly horizontal line suggests absence of association between urea and HbA1c
#points also do not really follow the line...
print(model1.summary())
#Adjusted R square suggests that 14.6% of variation in HbA1c can be explained by this model
#two variables age and urea w/ p<0.05 
# urea - negative association - declines with HbA1c in dataset
# age - positive association
#HDL was not significant, but is known to be decreased in diabetes subjects
#analysis is just a evidence, not a final conclusion as studies frequently have varying results


#binary logistic regression 
import statsmodels.api as sm

#include patients that are both T2DM and controls, map
log_data = data[data['CLASS'].isin(['Y', 'N'])].copy()
log_data['CLASS'] = log_data['CLASS'].map({'Y':1, 'N':0})

formula = 'CLASS ~ BMI + Gender +AGE + Urea + Cr'
logit_model = sm.Logit.from_formula(formula, data = log_data)
result = logit_model.fit()
print(result.summary())

#calculate fitted probabilities using the fitted model
fitted_probabilities = result.predict(log_data)

#plot logistic regression model
plt.figure(figsize = (10,6))
sns.scatterplot(x=log_data['BMI'], y=fitted_probabilities)
plt.xlabel('BMI')
plt.ylabel('Predicted Probability')
plt.title('Predicted Probabilities vs BMI')
plt.show()

#as BMI icnreases beyond 20-25, probability of T2DM increases rapidly 
#BMI is a significant predictor within our model...the only one <0.05
#coefficient of 0.70 --> subjects with diabetes have increased 0.789 units of BMI (kg/m3)
#results are adjusted for age and gender since they are included in the model

