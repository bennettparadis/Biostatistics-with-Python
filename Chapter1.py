# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn import datasets

iris = pd.read_csv(r"C:\Users\bparadis\Documents\Python Scripts\Biostatistics_python\iris.csv")

#difference between empty cells and Na or Nan - python reads as a non-numeric column; need to convert
#empty cells are converted to 'nan', but the column is kept as numeric
#all missing values should be converted to 'nan' so there is no accidental conversion of variables into strings
len(iris)

iris.isna().sum()

#coerce the data to numeric will also convert all invalid data to empty cells - all NAs are now 'nan'
iris['petal.length'] = iris['petal.length'].apply(pd.to_numeric,errors = 'coerce')

#EDA - exploratory/descriptive data analysis
variables = ['petal.width', 'petal.length', 'sepal.width', 'sepal.length']

#drop the missing or invalid data
cleandf=iris.dropna(subset=variables)
df = cleandf

#if needing to replace the name of a categorical factor use replace()
df2 = df.replace('Virginica', 'new_name', regex = True)

#drop a column
#df2 = df2.drop('column_name')

#descriptive statistics
df.describe()

#descriptive stats for each category in df
dtable=df.groupby(['variety']).describe()

#transpose
dtable=dtable.transpose()

#visualize the iris data and understand the distribution of the variables
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns
plot.figure()
df[variables].hist()
#distributions may appear unclear -- recall that there are multiple species in the dataframe, so might be
#best to separate them to understand distribution of the data better

#scatterplotting
plot.figure()
sns.scatterplot(x=df["sepal.length"],
                     y = df["sepal.width"],
                     hue=df['variety'])