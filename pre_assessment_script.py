import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

### 1.  Loading data into Pandas
## Import raw data from GitHub
url = 'https://raw.githubusercontent.com/danehamlett/UC_Davis/master/Price_vs_Sales.csv'
df = pd.read_csv(url)

## View raw data
print(df)


### 2.  Writing an apply function that transforms a column
### Transform Sales column for linear regression modeling
## Alternative Method: df['Log_Sales'] = np.log10(df['Sales'])
df['Log_Sales'] = df.apply(lambda row: np.log10(row.Sales), axis=1)

## View data
print(df)


### 3.  A basic Data Visualization using Seaborn or Plotly library or Matplotlib
### Using matplotlib, show a basic visualization (scatter plot)

## X and Y values
x = df['Price']
y = df['Sales']

## Render the chart
plt.plot(x, y, '.')
plt.show()


### 4.  Some type of Machine Learning technique on the data - Linear Regression
## X and Transformed Y values
x = df['Price']
y_log = df['Log_Sales']

## Render the chart
plt.plot(x, y_log, '.')

## Create a scatter plot with a linear regression trend line
m, b = np.polyfit(x, y_log, 1)
plt.plot(x, m*x + b, '-')
plt.show()

## Run a regression model
x = sm.add_constant(x, prepend=True)
results = smf.OLS(y_log,x).fit()
print(results.summary())
