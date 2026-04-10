# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 13:51:17 2025

@author: bparadis
"""
#biostatistical inference using hypothesis tests & effect sizes
#t-test, Wilcoxon rank test, chi-squared test, Pearsons correlation test, ANOVA, and Kruskal-Wallist test
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#load data & isolate diabetic patients
data=pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Dataset of Diabetes.csv")
data = data[data['CLASS'] == 'Y']

#comparing M & F diabetic cases for HDL
data1 = data[data['Gender'] == 'F']
data2 = data[data['Gender'] == 'M']

#select HDL
HDLf = data1['HDL']
HDLm = data2['HDL']

#generate summary statistics for male/female diabetic patients
summary_HDLf = HDLf.describe()
summary_HDLm = HDLm.describe()
summary_df = pd.DataFrame({
    'Female HDL Summary': summary_HDLf,
    'Male HDL Summary': summary_HDLm})

#how large is the difference in magnitude and is it significant?
#calculate Cohen's d -- measure magnitude (strength of a difference/effect) to determine effect size
cohens_d = (np.mean(HDLf) - np.mean(HDLm)) / (np.sqrt((np.std(HDLf)**2 + np.std(HDLm)**2 ) / 2))
print(cohens_d)

#t-test -- statistical significance between groups
t_statistic, p_value = stats.ttest_ind(HDLf, HDLm)
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")
#statistically the difference is significant; but the Cohen d suggests small effect size

#plot a histogram of the variables - visualize the small effect size

plt.figure(figsize = (10,6))
plt.hist(HDLf, bins = 20, color = 'blue', alpha=0.7, label = 'Female HDL')
plt.hist(HDLm, bins = 20, color = 'orange', alpha=0.7, label = 'Male HDL')

mean_HDLf = np.mean(HDLf)
mean_HDLm = np.mean(HDLm)

st_HDLf = np.std(HDLf)
st_HDLm = np.std(HDLm)

plt.axvline(mean_HDLf, color = 'blue', linestyle = 'dashed',
            linewidth=2, label = 'Female Mean')
plt.axvline(mean_HDLm, color = 'orange', linestyle = 'dashed',
            linewidth=2, label = 'Male Mean')

plt.legend()
plt.xlabel('HDL Values')
plt.ylabel('Frequency')
plt.title('Histograms of HDL Values by Gender with Mean Lines')
plt.show()


#could use a non-parametric test, one that does not have a normality assumption, and check significance
#Wilcoxon rank sum test (aka Mann-Whitney U test)

from scipy.stats import ranksums

statistic, p_value = ranksums(HDLf, HDLm)
print("Mann-Whitney U statistic:", statistic)
print("P-value:", p_value)
#still statistically significant difference between M & F



#new comparison between diabetics vs non-dabetics based on TG
data=pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Dataset of Diabetes.csv")

data1 = data[data['CLASS'] == 'Y']
data2 = data[data['CLASS'] == 'N']

#diabetic triglyceride levels
TGd = data1['TG']
#nondiabetic triglyceride levels
TGnd=data2['TG']

#calculated Cohen's d
cohens_d = (np.mean(TGd) - np.mean(TGnd)) / (np.sqrt((np.std(TGd)**2 + np.std(TGnd)**2 ) / 2))
print(cohens_d)
t_statistic, p_value = stats.ttest_ind(TGd, TGnd)
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

#summary stats
summary_TGd = TGd.describe()
summary_TGnd = TGnd.describe()

summary_df = pd.DataFrame({
    'Diabetes TG Summary': summary_TGd,
    'Control TG Summary': summary_TGnd})
print(summary_df)


#chi square test is often used for categorical variables - ex./ whether lipid levels are normal (<1.7mmol/L) or abnormal (>1.7)
#use data from previous steps, make a new column 'increasedtg' for categorical variable
#then can compare TG levels for diabetic and non diabetic patients with respect to a clinically defined threshold
data['increasedtg'] = data['TG'].apply(
    lambda x: 
        'yes' if x>= 1.7
        else 'no')
# alternative method -- data['increasedtg'] = np.where(data['TG'] >= 1.7, 'yes', 'no')

from scipy.stats import chi2_contingency
import researchpy as rp

#get defined patients (y or n for diabetic)
filtered_data = data[data['CLASS'].isin(['Y', 'N'])].copy()

#create contingency table
contingency_table = pd.crosstab(
    filtered_data['CLASS'], filtered_data['increasedtg'])

#convert frequencies to ratios
contingency_table_ratios = contingency_table.apply(
    lambda r: r/r.sum(), axis = 1)

#71% of diabetics have elevated TGs compared to 34% of non-diabetic patients
#perform chi-square test to evaluate significance
chi2, p, dof, expected = chi2_contingency(contingency_table)

#Cramer's Phi - a measure of association between two categorical variables in a contingency table; 
#it adjusts for sample size and table dimensions, returns a value 0-1 with 1 equalling a perfect association b/w variables
n = contingency_table.values.sum() #total number of observations
min_dim = min(contingency_table.shape) - 1 #min dimension minus one
cramers_phi = (chi2 / (n * min_dim))**0.5

#print results
print(f"Chi-squared value: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"Cramer's Phi: {cramers_phi:.4f}")
print(f"\nContingency Table Ratios:\n{contingency_table_ratios}")
#highly statistical significance, but moderate association based on Cramer's scale



#Pearson correlation method - testing for multiple associations between continuous variables 
#drop non-continuous/ID columns from filtered dataset
filtered_data = filtered_data.iloc[:,2:]

#compute correlation matrix
corr_matrix = filtered_data.corr(
    method = 'pearson', numeric_only = True)

#format into a readable csv table for output
#compute p-values for the correlation matrix
filtered_data = filtered_data.select_dtypes(include = 'number')
corr_matrix = filtered_data.corr(method = 'pearson')

p_values = pd.DataFrame(
    index = corr_matrix.index, columns=corr_matrix.columns)

for i in range(len(corr_matrix)):
    for j in range(i, len(corr_matrix)):
        r, p = stats.pearsonr(filtered_data.iloc[:,i],
                              filtered_data.iloc[:,j])
        p_values.iloc[i,j] = p
        p_values.iloc[j,i] = p 
        
#display correlation matrix with p-valuyes results marked w/ asterisk
display_matrix = corr_matrix.applymap(lambda x: f"{x:.2f}")
for i in range(len(p_values)):
    for j in range(len(p_values)):
        if p_values.iloc[i,j] < 0.05:
            display_matrix.iloc[i,j] += "*"
print(display_matrix)

display_matrix.to_csv(r'C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Correlation_Matrix.csv', index=True)
 

#makign a correlation plot for visual analysis
import seaborn as sns

#mask for the upper triangle of matrix (skip redundancies of matrix)   
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

#custom colormap to highlight significant p-values
cmap = sns.diverging_palette(230, 20, as_cmap=True)

#an array with correlation coefficients and asterisks for signif p-values
annot_array = np.vectorize(
    lambda x, y: f"{corr_matrix.iloc[x, y]:.2f}" +
    ("*" if p_values.iloc[x, y]<0.05 else "")
) (
       np.arange(len(corr_matrix)),
       np.arange(len(corr_matrix)) [:, None]
     )

#create a heatmap with correlation coefficients and signif p-values
sns.heatmap(corr_matrix, mask = mask, cmap=cmap,
            annot = annot_array, fmt='s')




#ANOVA & Kruskal-Wallist tests - analyzing multiple groups within variables 
#ANOVA comparing lipid levels of various BMI subjects
data=pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Dataset of Diabetes.csv")
filtered_data = data[data['CLASS'].isin(['Y', 'N'])]
       
#create new column for weight class based on BMI
def weight_class(bmi): 
    if bmi < 25:
        return 'normal'
    elif bmi >= 25 and bmi < 30:
        return 'overweight'
    else:
        return 'obese'
    
filtered_data['weight_class'] = filtered_data['BMI'].apply(weight_class)           
       
#create a table for descriptive stats of each class
normal_stats = filtered_data[
    filtered_data['weight_class'] == 'normal']['TG'].describe()
                                              
overweight_stats = filtered_data[
    filtered_data['weight_class'] == 'overweight']['TG'].describe()                                           
                                              
obese_stats = filtered_data[
    filtered_data['weight_class'] == 'obese']['TG'].describe()

stats_table = pd.DataFrame({'Normal':normal_stats, 'Overweight':overweight_stats, 'Obese':obese_stats})
print(stats_table)

#run ANOVA on TG for the three classes - assumes normality of the data, compares means
fvalue, pvalue = stats.f_oneway(
    filtered_data[filtered_data['weight_class'] == 'normal']['TG'], 
    filtered_data[filtered_data['weight_class'] == 'overweight']['TG'],
    filtered_data[filtered_data['weight_class'] == 'obese']['TG'],
    )
results = pd.DataFrame({'F-value': [fvalue],
                        'p-value': [pvalue]})

print(results)

#save output tables as csv files
stats_table.to_csv(
    r'C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Descriptive_bmi_stats.csv',
    index = True)
results.to_csv(
    r'C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\ANOVA_results.csv', 
    index = True)



# alternatively, can use non-parametric test that does not assume normality of the data and compare the medians
# Kruskal-Wallis test
data=pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Dataset of Diabetes.csv")
filtered_data = data[data['CLASS'].isin(['Y', 'N'])]

#create new column for weight class based on BMI
def weight_class(bmi): 
    if bmi < 25:
        return 'normal'
    elif bmi >= 25 and bmi < 30:
        return 'overweight'
    else:
        return 'obese'
    
filtered_data['weight_class'] = filtered_data['BMI'].apply(weight_class) 

fig, ax = plt.subplots(figsize = (12,9))

#specify order of categories
order = ['normal', 'overweight', 'obese']

#new dataframe with sorted rows
sorted_data = filtered_data.copy()
sorted_data['weight_class'] = pd.Categorical(
    sorted_data['weight_class'], categories=order,
    ordered=True)
sorted_data.sort_values(by="weight_class", inplace= True)

#create boxplot
sorted_data.boxplot(column = 'TG', by= 'weight_class', ax=ax)
plt.title('Boxplots of TG by Weight Class')
plt.suptitle('')

#Perofrm Kruskal-Wallis test
hvalue, pvalue = stats.kruskal(
    sorted_data[sorted_data['weight_class'] == 'normal']['TG'], 
    sorted_data[sorted_data['weight_class'] == 'overweight']['TG'],
    sorted_data[sorted_data['weight_class'] == 'obese']['TG'],
    )

#add test results to plot
plt.annotate(
    f'Kruskal-Walls H-value: {hvalue:.2f}\np-value: {pvalue:.2e}',
    xy = (0.7,0.9), xycoords='axes fraction')

plt.show()

#green lines show the median of each group
#KW test is statistically significant
#there is a small increase in TG across weight classes, but not large in magnitude



















