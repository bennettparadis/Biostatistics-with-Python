# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:47:30 2026

@author: bparadis
"""
#data source: http://archive.ics.uci.edu/dataset/45/heart+disease
#citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.

#set column names according to documentation
col_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
             "exang", "oldpeak", "slope", "ca", "tha1", "cad"]

#load libraries & dataset w/col names
import pandas as pd
import numpy as np

dataset = pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\heart_disease\processed.cleveland.data",
                      names = col_names, header=None, na_values=["?"])

#explore, verify structure
print(dataset.head())

#goal of analysis is to compare patients with and without Coronary artery disease (CAD)
#default CAD variable is on a scale 0-4 w/ classification on severity of cad
#re-code to binary classification
dataset['cad'] = np.where(dataset['cad'] > 0, 1, 0)

#group dataset by CAD status and generate descriptive statistics - find variables of possible significance
grouped_data = dataset.groupby("cad")
statistics = grouped_data.describe()
transposed_statistics = statistics.T
print(transposed_statistics)

#exploratory analysis
#in many clinical research projects, age is often the first aspect explored 
#descriptive statistics shows a ~4 year diff between CAD and non-CAD subjects
#perform t-test to determine if its a statistically signif difference
cad_age = dataset[dataset['cad']==1]['age']
control_age = dataset[dataset['cad']==0]['age']

from scipy import stats
t_statistic, p_value = stats.ttest_ind(cad_age, control_age)
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

#since there is a statistically significant difference in age between subjects,
#age should be included as a variable so as to avoid any confounding bias in future analyses

#other variables of significance - ST depression (oldpeak) & ST slope
#check for statistical significance between CAD and control groups
datacad=dataset[dataset['cad']==1]
datacontrol=dataset[dataset['cad']==0]
cadst=datacad['oldpeak']
contst=datacontrol['oldpeak']
t_statistic, p_value = stats.ttest_ind(cadst, contst)
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

#increased max heart rate in control is bc of age, not CAD

#Research questions
#1 - is ST depression magnitude associated with max heart rate during stress test?
#2- is ST depression mag associated with CAD?
#3- is exercise induced angina related to CAD?

#Question 1 - is ST depression magnitude associated with max heart rate during stress test?
#max heart rate is thalach in df
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

data = dataset
data_filtered = data[data['cad']==1]
linmod = smf.ols(
    formula = 'thalach ~ oldpeak + age ', data=data_filtered
    ).fit()
print(linmod.summary())

#plot regression line w/ confidence interval
sns.regplot(x='oldpeak', y='thalach', data = data_filtered, ci =95)
plt.title('Linear Regression: Stress test maximum bpm vs ST')
plt.xlabel('ST depression level (mm)')
plt.ylabel('Stress test max bp')
plt.show()

#results - model explains very little variance (adj R2 of 0.046)
# bc ST depression has a low p-value, it is likely a significant
# predictor & is associated, but is not the dominant predictor


#Question 2 - are ST depression, age, and cholesterol associated with CAD?
# logistic regression
import statsmodels.api as sm

formula = 'cad ~ oldpeak + age + chol'
logit_model = sm.Logit.from_formula(formula, data = dataset)
results = logit_model.fit()
print(results.summary())

#psuedo R-sq equivalent of R-sq for logistic model
#this model explains more of the variability
#age and ST are both significant
#CI for age suggests that bc the lower bound is close to 0, then
#less confident about the magnitude of the effect
#for ST depression, it's the opposite, so more confident
#in the magnitude of effect bc higher lower bound

#Question 3 - add exercise induced angina (exang in df)
formula = 'cad ~ oldpeak + age + chol + exang'
logit_model = sm.Logit.from_formula(formula, data = dataset)
results = logit_model.fit()
print(results.summary())

#exang improves the model, as it is a significant predictor
#ST depression is also a credible predictor

#for logistic regression, we are interpreting the probability of
#CAD. but the coefficients need to be adjusted to a probability 
#scale using odds-ratios for reporting

#logistic regression coefficients are log values of odds-ratios
#need to convert to OR scale by exponentiate the coefficients (opposite of log)

#calculate odds ratio scale by exponentiating
odds_ratios = np.exp(results.params)
conf_intervals = np.exp(results.conf_int())
p_values = results.pvalues

#combine odds ratios, confidence intervals, and p-values into a dataframe
summary_df = pd.DataFrame({
    'Odds Ratio': odds_ratios,
    'CI Lower': conf_intervals[0],
    'CI Upper': conf_intervals[1],
    'P-value': p_values})

print("\nSummary with Odds Ratios, CIs, and p-values:")
print(summary_df)
summary_df.to_csv('CAD_logistic_results.csv', index=True)

#interpretation
#an increase in 1 mm of ST depression (oldpeak) is associated with an increase
#of odds-ratio from 1 to 2.09
#presence of exercise inducde angina (exang) is associated with 5.65 times 
#increased odds of CAD




