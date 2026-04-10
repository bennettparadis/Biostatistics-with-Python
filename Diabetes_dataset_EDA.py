# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 11:27:24 2025

@author: bparadis
"""

import pandas as pd
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns

data=pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\Dataset of Diabetes.csv")

#validating and wrangling the dataset

#check for missing values -- none in this dataset
data.isna().sum()

#remove unneccessary variables
df=data
df=df.drop(['ID', 'No_Pation'], axis = 1)

#remove P class -- predicted diabetes cases; only want diagnostically confirmed cases for analysis
df=df[df["CLASS"].str.contains("P") == False]

#transpose and organize table by gender -- summary statistics for each numerical variable by gender
dtable1 = df.groupby(['Gender']).describe()
dtable1 = dtable1.transpose()

#3 classes present due to upper/lower case typo - quick fix
df = df.replace('f', 'F', regex=True)
dtable1 = df.groupby(['Gender']).describe()
dtable1 = dtable1.transpose()

#quick notes comparing the genders and evaluating the dataset
# AGE - mean and median are comparable between M & F

#group by class to double check data - Y / N for diabetes
dtable2 = df.groupby(['CLASS']).describe()
dtable2 = dtable2.transpose()

#spaces in the class names, resulting in more classes than expected - remove w/ strip()
df['CLASS'] = df['CLASS'].str.strip()
dtable2 = df.groupby(['CLASS']).describe()
dtable2 = dtable2.transpose()

#can easily explore the variables and compare between diabetic and nondiabetic patients
#HbA1c has a median and avg 2x among diabetic cases, making it a key biomarker

#group with multiple variables 
table_final = df.groupby(['Gender', 'CLASS']).describe()
table_final = table_final.transpose()
#Urea is highly variable in F diabetic patients (high std); avg in M & F is slightly higher in diabetic patients

#visualizing data
#isolate four variables of interest
variables = ['TG', 'LDL', 'HDL', 'HbA1c']

#take a look at the distribution of Hb1Ac - glycated hemoglobin (how much is glycated over 3 months)
sns.displot(data = df[variables], x ="HbA1c", hue =df["CLASS"], kind = "kde")
#y-axis is kernel density estimate 
#peaks and distributions are different, HbA1c values are higher for diabetic patients

#plot for triglycerides
sns.displot(data = df[variables], x ="TG", hue =df["CLASS"], kind = "kde")
#peaks are aligned, but more variability among diabetics beyond value of 2

#plot for LDL
sns.displot(data = df[variables], x ="LDL", hue =df["CLASS"], kind = "kde")

#plot for HDL
sns.displot(data = df[variables], x ="HDL", hue =df["CLASS"], kind = "kde")


#stacked histogram plots for visualizing distribution
#HbA1c
sns.displot(data=df[variables], x ='HbA1c', hue = df["CLASS"], multiple = "stack")

#LDL - confirm that the distribution is similar across controls and diabetics, minor differences toward upper values of LDL
#difference between classes is uniform/similar across LDL
sns.displot(data=df[variables], x ='LDL', hue = df["CLASS"], multiple = "stack")

#TG - difference between control and diabetics increases as the TG increases (few controls beyond 4)
sns.displot(data=df[variables], x ='TG', hue = df["CLASS"], multiple = "stack")

#HDL - little differences, few occurences of high HDL levels in either class
sns.displot(data=df[variables], x ='HDL', hue = df["CLASS"], multiple = "stack")

#takeaways from distribution plots - largest difference was in the HbA1c variable, TG somewhat beyond a level of 4

#scatterplots for bivariate analysis - small grouping for controls with low HbA1c and low BMI values
sns.scatterplot(x = df["BMI"], 
                y = df["HbA1c"],
                hue = df["CLASS"])

#jointplot combines distribution plot plus scatterplot visualization
sns.jointplot(x = df["BMI"], 
                y = df["HbA1c"],
                hue = df["CLASS"])

#boxplot for a closer look at HbA1c 
sns.boxplot(x=df["Gender"],
            y = df["HbA1c"],
            hue = df["CLASS"])












