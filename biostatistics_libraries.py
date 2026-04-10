# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 11:46:04 2025

@author: bparadis
"""
#intro to libraries used for hypothesis testing
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#load data
data=pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Dataset of Diabetes.csv")

#isolate data with just confirmed cases of diabetes
data = data[data['CLASS'] == 'Y']

#comparing M & F diabetic cases for HDL
data1 = data[data['Gender'] == 'F']
data2 = data[data['Gender'] == 'M']

#select HDL
HDLf = data1['HDL']
HDLm = data2['HDL']

#make plot to compare HDL between M & F diabetic patients
labels = ['HDLf', 'HDLm']
heights = [HDLf.mean(), HDLm.mean()]
plt.bar(labels, heights)
plt.xlabel('Gender')
plt.ylabel('HDL')
plt.title('Mean HDL levels by Gender')
plt.show()

#perform independent t-test 
t_statistic, p_value = stats.ttest_ind(HDLf, HDLm)

print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")
#results show a significant difference between genders in diabetic subjects


#test for diff with LDL
LDLf = data1['LDL']
LDLm = data2['LDL']

#make plot to compare HDL between M & F diabetic patients
labels = ['LDLf', 'LDLm']
heights = [LDLf.mean(), LDLm.mean()]
plt.bar(labels, heights)
plt.xlabel('Gender')
plt.ylabel('LDL')
plt.title('Mean LDL levels by Gender')
plt.show()

#perform independent t-test 
t_statistic, p_value = stats.ttest_ind(LDLf, LDLm)

print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")
#not a statistically significant difference between the groups for LDL
#"results weren't conclusive"


#chi-squared test -- count data
from scipy import chisquare
#sample data
observed = [10, 20]
expected = [15, 15]

chi2_statistic, p_value = chisquare(observed, f_exp= expected)
print(f"chi2-statistic: {chi2_statistic}")
print(f"p-value: {p_value}")


#predictive statistics - how much association can be identified among different variables
import statsmodels.api as sm

#among cases of patients with diabetes from main dataset, interested in Chol and TG
#ordinary least squares (OLS) - evaluate the linear relationship between independent variables
# and a target variable by finding the line that best fits the data based on squared differences
# between observed and predicted values. TG is dependent, Chol is independent in this case
Chol = data['Chol']
TG = data['TG']

#add a constant to the predictor variable
X = sm.add_constant(Chol)

#fit the regression model
model = sm.OLS(TG,X)
results = model.fit()
print(results.summary())
#R-squared - coefficient of determination - 11.2% of TG variation can be explained by cholesterol variation
#p-value for coeff suggests statistical signifc

#plotting results 
import seaborn as sns

#place variables into a new dataframe
df = pd.DataFrame({'Chol': Chol, 'TG': TG})
sns.lmplot(x='Chol', y = 'TG', data= df)
plt.xlabel('Chol')
plt.ylabel('TG')
plt.title('Regression Analysis: Chol vs TG')
plt.show()













