import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# had to install scikit learn package to use sklearn 
# --break-system-packages (external enviorment) bipass 
# correct way to use pkgs is Virtualized Enviorment (python built in feature) and install packages in there


# Load in dataset 
# Make wine dataframe which is the read csv file , has weird semicolon seperators specify with sep
wine_df = pd.read_csv('winequality-white2.csv', sep =';')
# create a list of features we want to use in our regression 
regression_list = ["alcohol", "pH","sulphates"]
outcome_variable = wine_df["quality"]
# regression dataframe is our wine dataframe but restricted to the regression list 
regression_df = wine_df[regression_list]

# linear regression model
linear_regression_model = LinearRegression()
# fit regression dataframe and our outcome variable 
linear_regression_model.fit(regression_df, outcome_variable)
# multiple coefficients , one coefficient per variable
model_ms = linear_regression_model.coef_
# still only have one intercept 
model_b = linear_regression_model.intercept_
print(f"Model Coefficients:{model_ms}\nModel Intercept:{model_b}")

# new regression model : take the regression list and multiply it by model coefficients (alcohol multiplied by coefficient ~.3, ph multiplied by ~.2...)
# then add  intercept to arrive at final quality 


