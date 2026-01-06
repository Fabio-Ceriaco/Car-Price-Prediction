# Car Price Prediction using Machine Learning
# Dataset: Car details v3.csv
# Objective: Predict the selling price of a car based on various features
# This project involves data preprocessing, exploratory data analysis (EDA), model training, and evaluation using Linear Regression and Lasso Regression models.
# The dataset contains various features such as car name, year of manufacture, selling price, present price, kilometers driven, fuel type, seller type, transmission type, owner type, mileage, engine capacity, max power, and torque.
# The selling price is the target variable to be predicted.
# The models will be evaluated based on metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared score.
# The goal is to build a robust model that can accurately predict car prices based on the provided features.


# Work Flow
# 1. Import Libraries
# 2. Load Data
# 3. Exploratory Data Analysis (EDA)
# 4. Data Preprocessing
# 5. Split Data
# 6. Model Training
# 7. Model Evaluation
# 8. Maker Prediction

# Machine Learning: Linear & Lasso Regression models


# =============================================================================================#
# 1. Import Libraries
# =============================================================================================#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================================#
# 2. Load Data
# =============================================================================================#

# Pandas Options

pd.set_option("display.max_columns", 13)


data_df = pd.read_csv("Car details v3.csv")


# =============================================================================================#
# 3. Exploratory Data Analysis (EDA)
# =============================================================================================#

# Print first 5 rows of the dataset
print(data_df.head())


# Print dataset Shape
print(data_df.shape)


# Print dataset Info
print(data_df.info())

# Print dataset Description
print(data_df.describe())

# Check for missing values
print(data_df.isnull().sum())

# Drop rows with missing values
data_df.dropna(inplace=True)

# Check for missing values again
print(data_df.isnull().sum())


# =============================================================================================#
# 4. Data Preprocessing
# =============================================================================================#


# Convert the values in selling_price units and to float type
data_df["selling_price"] = data_df["selling_price"] / 100000
data_df["selling_price"] = data_df["selling_price"].astype(float)


# Print distribution values in categorical columns
categorical_columns = ["fuel", "seller_type", "transmission", "owner", "torque"]
for col in categorical_columns:
    print("\n=============================================================")
    print(f"Unique values in {col}: {data_df[col].value_counts()}")
    print("\n=============================================================\n")

# Encoding fuel, seller_type, transmission, owner columns

data_df["fuel"] = data_df["fuel"].map(
    {"Petrol": 0, "Diesel": 1, "CNG": 2, "LPG": 3, "Electric": 4}
)
data_df["seller_type"] = data_df["seller_type"].map(
    {"Individual": 0, "Dealer": 1, "Trustmark Dealer": 2}
)
data_df["transmission"] = data_df["transmission"].map({"Manual": 0, "Automatic": 1})
data_df["owner"] = data_df["owner"].map(
    {
        "First Owner": 0,
        "Second Owner": 1,
        "Third Owner": 2,
        "Fourth & Above Owner": 3,
        "Test Drive Car": 3,
    }
)


# Taking letters off from engine and mileage columns

data_df["mileage"] = data_df["mileage"].str.split(" ").str.get(0).astype(float)

data_df["engine"] = data_df["engine"].str.split(" ").str.get(0).astype(int)

data_df["max_power"] = data_df["max_power"].str.split(" ").str.get(0).astype(float)


# Check data types after conversion
print(data_df.dtypes)

# Print first 5 rows after changes
print(data_df.head())


# Print unique values in categorical columns
categorical_columns = ["fuel", "seller_type", "transmission", "owner"]
for col in categorical_columns:
    print("\n=============================================================")
    print(f"Unique values in {col}: {data_df[col].value_counts()}")
    print("\n=============================================================\n")

# Check data types after conversion
print(data_df.dtypes)

# Drop torque column as it has too many unique values and we dont need it for prediction
data_df.drop(columns=["torque"], inplace=True)

# print first 5 rows after changes
print(data_df.head())

# Correlation between features and target variable


# 1. Check column names
print(data_df.columns)

# 2. Correlation matrix
data_df.drop(
    columns=["name"], inplace=True
)  # Dropping 'name' column for correlation analysis
print(data_df.corr())

# 3. Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =============================================================================================#
# 5. Split Data
# =============================================================================================#

X = data_df.drop(columns=["selling_price"], axis=1)
y = data_df["selling_price"]


print(X)
print(y)


# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)

# =============================================================================================#
# 6. Model Training
# =============================================================================================#

# Linear Regression Model

lin_reg_model = LinearRegression()

lin_reg_model.fit(X_train, y_train)

# Lasso Regression Model

lasso_reg_model = Lasso()

lasso_reg_model.fit(X_train, y_train)

# =============================================================================================#
# 7. Model Evaluation
# =============================================================================================#

# Linear Regression Model Evaluation

train_prediction = lin_reg_model.predict(X_train)
test_prediction = lin_reg_model.predict(X_test)

print("Linear Regression Model Evaluation:")
print("=============================================================")
print("Training Set Evaluation:")
print("MAE:", mean_absolute_error(y_train, train_prediction))
print("MSE:", mean_squared_error(y_train, train_prediction))
print("RMSE:", np.sqrt(mean_squared_error(y_train, train_prediction)))
print("R2_Score:", r2_score(y_train, train_prediction))

# Conclusions:
# MEA : 2.758162640202307 (Mean Absolute Error) - The average absolute difference between predicted and actual values is approximately 2.76 (in 100000$ units).
# MSE : 21.895003798743097 (Mean Squared Error) - The average squared difference between predicted and actual values is approximately 21.90 (in 100000$ units squared).
# RMSE: 4.679209740836918 (Root Mean Squared Error) - The standard deviation of the prediction errors is approximately 4.68 (in 100000$ units).
# R2_Score: 0.671084955075824 (R-squared Score) - The model explains approximately 67.11% of the variance in the target variable (selling price).

# Visualize the actual vs predicted values for training set
plt.scatter(y_train, train_prediction, color="blue")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price (Training Set)")
plt.show()

# As we can see from the scatter plot, the predicted values are closely aligned with the actual values,
# indicating a good fit of the model on the training data.


print("=============================================================")
print("Testing Set Evaluation:")
print("MAE:", mean_absolute_error(y_test, test_prediction))
print("MSE:", mean_squared_error(y_test, test_prediction))
print("RMSE:", np.sqrt(mean_squared_error(y_test, test_prediction)))
print("R2_Score:", r2_score(y_test, test_prediction))

# Conclusions:
# MEA: 2.767179560354457 - The average absolute difference between predicted and actual values is approximately 2.77 (in 100000$ units).
# MSE: 21.225346861146075 - The average squared difference between predicted and actual values is approximately 21.23 (in 100000$ units squared).
# RMSE: 4.607097444285944 - The standard deviation of the prediction errors is approximately 4.61 (in 100000$ units).
# R2_Score: 0.671639463130064 - The model explains approximately 67.16% of the variance in the target variable (selling price).

# Visualize the actual vs predicted values for training set
plt.scatter(y_test, test_prediction, color="blue")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price (Testing Set)")
plt.show()

# Evaluation of Lasso Regression Model
train_prediction_lasso = lasso_reg_model.predict(X_train)
test_prediction_lasso = lasso_reg_model.predict(X_test)

print("Lasso Regression Model Evaluation:")
print("=============================================================")
print("Training Set Evaluation:")
print("MAE:", mean_absolute_error(y_train, train_prediction_lasso))
print("MSE:", mean_squared_error(y_train, train_prediction_lasso))
print("RMSE:", np.sqrt(mean_squared_error(y_train, train_prediction_lasso)))
print("R2_Score:", r2_score(y_train, train_prediction_lasso))

# Conclusions:
# MAE: 2.8926684236619575 - The average absolute difference between predicted and actual values is approximately 2.89 (in 100000$ units).
# MSE: 24.32245617392669 - The average squared difference between predicted and actual values is approximately 24.32 (in 100000$ units squared).
# RMSE: 4.93178022360351 - The standard deviation of the prediction errors is approximately 4.93 (in 100000$ units).
# R2_Score: 0.634618845529835 - The model explains approximately 63.46% of the variance in the target variable (selling price).

plt.scatter(y_train, train_prediction_lasso, color="blue")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price (Training Set)")
plt.show()

print("=============================================================")
print("Testing Set Evaluation:")
print("MAE:", mean_absolute_error(y_test, test_prediction_lasso))
print("MSE:", mean_squared_error(y_test, test_prediction_lasso))
print("RMSE:", np.sqrt(mean_squared_error(y_test, test_prediction_lasso)))
print("R2_Score:", r2_score(y_test, test_prediction_lasso))

# Conclusions:
# MAE: 2.927433709205544 - The average absolute difference between predicted and actual values is approximately 2.93 (in 100000$ units).
# MSE: 23.561570326768113 - The average squared difference between predicted and actual values is approximately 23.56 (in 100000$ units squared).
# RMSE: 4.854026197577441 - The standard deviation of the prediction errors is approximately 4.85 (in 100000$ units).
# R2_Score: 0.6354975995158563  - The model explains approximately 63.55% of the variance in the target variable (selling price).

plt.scatter(y_test, test_prediction_lasso, color="blue")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price (Testing Set)")
plt.show()
