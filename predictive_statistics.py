# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 11:35:43 2026

@author: bparadis
"""

#Predictive statistics 

#Linear regression for biostatistics - for predicting continuous variables
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

#load the data
data=pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Dataset of Diabetes.csv")
data = data.replace('f', 'F', regex =True)

#create a simple model
#Model1: univariate linear regression with HbA1c and BMI
model1 = smf.ols(formula = 'HbA1c ~ BMI', data =data).fit()
print(model1.summary())

#plot the association between the variables w/ trend line
sns.regplot(x= 'BMI', y ='HbA1c', data = data, ci=95)
plt.show()



#Logistic regression - for predicting categorical variables
#filter data to include only rows with a defined class
filtered_data = data[data['CLASS'].isin(['Y', 'N'])]

#map Y to 1 and N to 0 for the CLASS variable
filtered_data['CLASS'] = filtered_data['CLASS'].map({'Y':1, 'N':0})

#define the formula for logistic regression
formula = 'CLASS ~ HbA1c'

#create the logistic regression model
import statsmodels.api as sm
logit_model = sm.Logit.from_formula(formula, data=filtered_data)

#fit the model & print results
results = logit_model.fit()
print(results.summary())

#converged True status means the model successfully optimized the fit between the data and model
#can then calculate the probabilities for the logistic function of HbA1c -- can then plot the model
fitted_probabilities = results.predict(filtered_data)

#plot
plt.figure(figsize = (10,6))
sns.scatterplot(
    x=filtered_data['HbA1c'], y = fitted_probabilities)
plt.xlabel('HbA1c')
plt.ylabel('Predicted Probability')
plt.title('Predicted Probabilities vs HbA1c')
plt.show()



#Multiple linear and logistic regressions
#Model 2: multivariate linear regression - HbA1c determined by BMI, Age, and Gender
model2 = smf.ols(formula = 'HbA1c ~ BMI + AGE + Gender', 
                 data = data).fit()
print(model2.summary())

#adjust r-squared value tells us what proportion of the variability of the target variable (HbA1c)
# can be explained by the predictors; adjusted value is best for multivariate analysis as it
# adjusts for the number of predictor variables
#here, 22.8% of the variation in HbA1c can be explained by variantion in Gender, Age, and BMI

#Gender has a p-value of 0.174 --> not a significant predictor of HbA1c
#BMI and Age appear to be though with p-values of 0.000

#regression coefficients
#BMI = 0.1625 and Age = 0.075 --> positive association (and statistically validated)

#age and gender are often included in biostatistics models -- termed 'adjusted' for gender and age 

#Model 3: multivariate linear regression - HbA1c determined by TG, LDL, HDL, BMI, Age, and Gender
model3 = smf.ols(formula = 'HbA1c ~ BMI + AGE + Gender + TG + LDL + HDL', 
                 data = data).fit()
print(model3.summary())

#Gender, LDL, and HDL are not significant as independent predictors 
#HOWEVER, they can always affect other significant predictors; it is a good idea not to exclude them from the model
#TG appears to be significant -- coefficient of 0.2753 

#multiple logistic regression
formula = 'CLASS ~ HbA1c + AGE + Gender + BMI + HDL + LDL'
logit_model = sm.Logit.from_formula(
    formula, data= filtered_data)
result = logit_model.fit()
print(result.summary())

#significant predictors --> bmi, LDL, HbA1c
#Age is not significant, but it is a clinically known relevant factor for Type II diabetes; this model
#tells us that even though it is not significant in this model, it interacts with other variables and
#should be included











