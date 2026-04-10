import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#load the data
data=pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Dataset of Diabetes.csv")

#separate data according to class & isolate variable of interest -- glycoside hemoglobin
HbA1c_d = data[data['CLASS'] == 'Y']['HbA1c']
HbA1c_c = data[data['CLASS'] == 'N']['HbA1c']

#calculate avg and sd for variable
avg_HbA1c_d = HbA1c_d.mean()
std_HbA1c_d = HbA1c_d.std()
avg_HbA1c_c = HbA1c_c.mean()
std_HbA1c_c = HbA1c_c.std()

#create barplot w/ error bars
categories = ['T2DM', 'Controls']
values = [avg_HbA1c_d, avg_HbA1c_c]
stdevs = [std_HbA1c_d, std_HbA1c_c]
plt.bar(categories, values, yerr=stdevs, capsize = 5)
plt.ylabel('HbA1c')
plt.title('Average HbA1c Values')
plt.show()

#two sample/sided t-test
#generate summary statistics
summary_d = HbA1c_d.describe()
summary_c = HbA1c_c.describe()
summary_df = pd.DataFrame({
    'Diabetes HbA1c Summary': summary_d,
    'Control HbA1c Summary': summary_c})

#calculate Cohen's d for magnitude of effect size
cohens_d = (np.mean(HbA1c_d) - np.mean(HbA1c_c)
    ) / (np.sqrt( 
        (np.std(HbA1c_d) ** 2 + np.std(HbA1c_c) ** 2) / 2)
        )
print(f"cohens d: {cohens_d}") #large effect size, aligns with biological knowledge that HbA1c is a biomarker to diagnosing Typ2 diabetes

#t-test results
t_statistic, p_value = stats.ttest_ind(HbA1c_d, HbA1c_c)
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")


#one sample t-test
from scipy.stats import ttest_1samp
#compare HbA1c values of subjects relative to a specific threshold (7%)
t_statistic, p_value = ttest_1samp(HbA1c_d, 7)
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

