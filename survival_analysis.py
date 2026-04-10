#might need to forge an anaconda environment for libraries to be compatible 
##conda install -c conda-forge lifelines scikit-survival numpy scipy pandas matplotlib jupyter

from sksurv.datasets import load_veterans_lung_cancer
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
import matplotlib.pyplot as plt

# load the veterans lung cancer dataset
#data is split into arrays. The response variable, survival is represented within two columns:
# a boolean (T/F) and the number of days passed when the follow up occurred 
data_x, data_y = load_veterans_lung_cancer()

#Calculate the Kaplan-Meier survival estimates -- pooled, both treatment pops
time, survival_prob = kaplan_meier_estimator(data_y["Status"], data_y["Survival_in_days"])

#plot the survival curves
plt.step(time, survival_prob, where = "post")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.show()

#create a boolean array for the treatment groups - assign True/False for categorical variable
#based on whether it matches the specified condition
treatment = data_x["Treatment"] == "test"

#Calculate the kaplan-meier survival estimates for the first treatment group
#boolean indexing --> give status values and days for patients where treament = True (test)
time_treatment, survival_prob_treatment = kaplan_meier_estimator(
    data_y["Status"][treatment], data_y["Survival_in_days"][treatment])

#Calculate the kaplan-meier survival estimates for the second treatment group
#boolean indexing --> gives data for patients wher treament = False (standard)
time_control, survival_prob_control = kaplan_meier_estimator(
    data_y["Status"][~treatment], data_y["Survival_in_days"][~treatment])

#plot the survival curves for the first treatment group
plt.step(time_treatment, survival_prob_treatment,
         where="post", label = "standard treatment")

#plot the survival curves for the second/control group
plt.step(time_control, survival_prob_control,
         where="post", label = "test drug")

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
plt.show()

#survival for both treatments appears to be low after 400 days, neither are effective

#use logrank test to test for statistical significance - compare entire trajectory
#similar to a 2 group t-test
#compare curves w/ hypothesis testing H0(survival for test and contro lare the same) vs
# Ha(survival is different between groups)

# set group indicator and perform logrank test
group_indicator = data_x.loc[:, 'Treatment']
chi2, pvalue = compare_survival(data_y, group_indicator)
print(chi2) #how far apart the curves are
print(pvalue) #probability of seeing this difference if curves were actually the same
#no statistical difference between the two groups


#look at breast cancer dataset 
from sksurv.datasets import load_breast_cancer

data_x, data_y = load_breast_cancer()
#data includes gene expressions, age, er, grade, and tumor size

#compare subgroups based on tumor size where 2cm is cutoff
print("Feature columns:", data_x.columns)
print("Target columns:", data_y.dtype.names)
event_col = 'e.tdm'
time_col = 't.tdm'

group_a = data_y[data_x["size"] >= 2] 
group_b = data_y[data_x["size"] < 2] 

#calculate survival for each group
time_a, surv_a = kaplan_meier_estimator(
    group_a[event_col], group_a[time_col])

time_b, surv_b = kaplan_meier_estimator(
    group_b[event_col], group_b[time_col])

#plot the KM curves
plt.step(time_a, surv_a, where ="post", label="Size >= 2") 
plt.step(time_b, surv_b, where ="post", label="Size < 2") 
plt.ylabel("est. probability of survival $\hat{S} (t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
plt.show()

#subgroup with smaller tumors has better survival probabilities, descends much more slowly
#logrank test to confirm w/ a non-parametric hypothesis test
group_indicator = (data_x['size']>=2).astype(int)
chi2, pval = compare_survival(data_y, group_indicator)
print(chi2)
print(pval)
#result is statistically significant


#Cox (proportional hazards) regression
#method used in survival analysis to evalutate the relationship between time until an event of interest occurs and predictor variables
#it can handle censoring (where the event has not occurred by the end of the study)
# and create multivariate models by adding other covariates

#Cox model is based on the hazard function -- instantaneous rate of failure at a given time
#hazard is assumed to be proportional across different levels of the predictor variables.

#Cox model assumption --> hazard ratio (HR) is constant over time for any individual over time

#output of Cox regression includes HR to quantify the relative risk of experiencing the event
#dor different levels of a predictor variable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.datasets import load_breast_cancer
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

#load breast cancer dataset
data_x, data_y = load_breast_cancer()

#col names
print("Feature columns:", data_x.columns)
print("Target columns:", data_y.dtype.names)

#replace cens and time with col names
event_col = 'e.tdm' 
time_col = 't.tdm'

#convert 'er' to categorical var
data_x['er'] = pd.Categorical((data_x['er']))

#create dummy variables
data_x = pd.get_dummies(data_x, drop_first = True)

#add a new coveriate based on the 'size' column
data_x['size_group'] = (data_x['size'] >= 2).astype(int)

#select only columns of interest for analysis
data_x = data_x[['age', 'er_positive', 'size_group']]

#prepare data for CoxPHFitter
df = data_x.copy()
df[time_col] = data_y[time_col]
df[event_col] = data_y[event_col]

#fit the CoxPHFitter model
cph = CoxPHFitter()
cph.fit(df, duration_col = time_col, event_col = event_col)
cph.print_summary()

#coefficients are slops of the regression line; they are converted to hazard
#ratios under the exp(coef) columns 
#value of 1 means no difference
#value of 2 means twice the increased hazard or risk

#interpreting the results:
#HR for age is 1.01 --> 1% increase in risk, nonsignificant w/ p=0.57
#HR for er_positive is almost a double decrease in risk
#   confidence intervals cross 1, rely on p-value for significance
#size_group --> tumor size >=2cm group has a 3x risk increase, stat signif

#confints --> if range is above 1, significant increase in risk; 
#if range is below 1, signif decrease in risk

#plot
from lifelines import KaplanMeierFitter

kmf=KaplanMeierFitter()

#fit the model and plot survival function w/ confints
for name, grouped_df in df.groupby('size_group'):
    kmf.fit(grouped_df[time_col], grouped_df[event_col],
            label='Size group ' + str(name))
    kmf.plot(ci_show=True)
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.show()






