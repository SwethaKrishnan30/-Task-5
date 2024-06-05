# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:01:52 2024

@author: Joshua
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data from CSV file
data = pd.read_csv("C:/Users/Joshua/Downloads/Advertising.csv")

# Displaying the first few rows of the dataframe
print(data.head())

# Plotting pairplots
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=7, aspect=0.7, kind='reg')
plt.show()

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=2)
plt.show()

# Defining the independent variables (TV, Radio, Newspaper) and the dependent variable (Sales)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Adding a constant to the independent variables
X = sm.add_constant(X)

# Building the model
model = sm.OLS(y, X).fit()

# Displaying the model summary
print(model.summary())

# Residuals
residuals = model.resid

# Plotting residuals
plt.figure(figsize=(10, 6))
plt.scatter(model.fittedvalues, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

# QQ plot
sm.qqplot(residuals, line='45')
plt.show()

# Calculating VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)
