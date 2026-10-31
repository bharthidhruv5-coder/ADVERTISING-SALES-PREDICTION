# =====================================================
# ADVERTISING SALES PREDICTION PROJECT
# =====================================================

# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as s

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error


# 2. LOAD DATASET
df = pd.read_csv(r"D:\ml\advertising_sales.csv")

print("\n========== DATASET PREVIEW ==========")
print(df.head())

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== DATASET SUMMARY ==========")
print(df.describe())

print("\nDataset Shape:", df.shape)


# 3. UNDERSTANDING DATA
print("\n========== NUMERICAL COLUMNS ==========")
print(df.select_dtypes(include="number").columns.tolist())

print("\n========== CATEGORICAL COLUMNS ==========")
print(df.select_dtypes(exclude="number").columns.tolist())

print("\n========== MISSING VALUES BEFORE ==========")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove Duplicate Rows
df = df.drop_duplicates().copy()

# Handle Missing Values
numeric_columns = df.select_dtypes(include="number").columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

categorical_columns = df.select_dtypes(exclude="number").columns

for column in categorical_columns:
    if df[column].isnull().any() and not df[column].mode().empty:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\n========== MISSING VALUES AFTER ==========")
print(df.isnull().sum())


# 4. OUTLIER DETECTION
print("\n========== NEWSPAPER OUTLIERS ==========")

Q1 = df["Newspaper"].quantile(0.25)
Q3 = df["Newspaper"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Newspaper"] < lower) |
    (df["Newspaper"] > upper)
]

print("Lower Bound:", lower)
print("Upper Bound:", upper)
print("Outlier Count:", len(outliers))

print(outliers[["ID", "Newspaper", "Sales"]])


# 5. BEFORE AND AFTER OUTLIER COMPARISON
rows_before = len(df)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.boxplot(y=df["Newspaper"])
plt.title("Before Outlier Removal")

# Remove Newspaper outliers using IQR
df = df[
    (df["Newspaper"] >= lower) &
    (df["Newspaper"] <= upper)
].copy()

plt.subplot(1, 2, 2)
sns.boxplot(y=df["Newspaper"])
plt.title("After Outlier Removal")

plt.tight_layout()
plt.show()

print("Rows Before:", rows_before)
print("Rows After:", len(df))


# 6. HISTOGRAM
df[["TV", "Radio", "Newspaper", "Sales"]].hist(
    figsize=(10, 8)
)

plt.suptitle("Advertising Dataset Histogram")
plt.tight_layout()
plt.show()


# 7. BOX PLOT
plt.figure(figsize=(8, 5))

sns.boxplot(data=df[["TV", "Radio", "Newspaper"]])

plt.title("Advertising Channels Box Plot")
plt.show()


# 8. BAR CHART
df[["TV", "Radio", "Newspaper"]].mean().plot(
    kind="bar",
    title="Average Advertising Budget"
)

plt.ylabel("Average Value")
plt.tight_layout()
plt.show()


# 9. PIE CHART
df[["TV", "Radio", "Newspaper"]].sum().plot(
    kind="pie",
    autopct="%1.1f%%",
    title="Advertising Budget Share"
)

plt.ylabel("")
plt.tight_layout()
plt.show()


# 10. CORRELATION MATRIX
plt.figure(figsize=(8, 6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# 11. AUTOMATED EDA REPORT
report = s.analyze(df)

report.show_html("sweetviz_report.html")

print("Sweetviz HTML Report Generated!")


# 12. FEATURE SELECTION
X = df[["TV", "Radio", "Newspaper"]]

y = df["Sales"]

print("\nIndependent Variables: TV, Radio, Newspaper")
print("Dependent Variable: Sales")


# 13. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# 14. LINEAR REGRESSION MODEL
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# 15. MODEL EVALUATION
linear_r2 = r2_score(y_test, y_pred)

linear_mse = mean_squared_error(y_test, y_pred)

linear_rmse = np.sqrt(linear_mse)

print("\n========== LINEAR REGRESSION ==========")

print("R2 Score:", linear_r2)
print("MSE:", linear_mse)
print("RMSE:", linear_rmse)


# 16. ACTUAL VS PREDICTED GRAPH
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.show()


# 17. RIDGE REGRESSION
ridge = Ridge()

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

ridge_r2 = r2_score(y_test, ridge_pred)

ridge_mse = mean_squared_error(y_test, ridge_pred)

ridge_rmse = np.sqrt(ridge_mse)

print("\n========== RIDGE REGRESSION ==========")

print("R2 Score:", ridge_r2)
print("MSE:", ridge_mse)
print("RMSE:", ridge_rmse)


# 18. MODEL COMPARISON
comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression"
    ],
    "R2 Score": [
        linear_r2,
        ridge_r2
    ],
    "MSE": [
        linear_mse,
        ridge_mse
    ],
    "RMSE": [
        linear_rmse,
        ridge_rmse
    ]
})

print("\n========== MODEL COMPARISON ==========")

print(comparison)


# 19. CROSS VALIDATION
cv_scores = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\n========== CROSS VALIDATION ==========")

print("CV Scores:", cv_scores)

print("Average CV R2:", cv_scores.mean())


# 20. FINAL RESULTS
print("\n========== FINAL PROJECT RESULTS ==========")

print("Dataset Records:", len(df))

print("Model: Linear Regression")

print("Target Variable: Sales")

print("R2 Score:", linear_r2)

print("RMSE:", linear_rmse)

print("Average Cross Validation R2:", cv_scores.mean())


# 21. PROJECT COMPLETED
print("\nAdvertising Sales Prediction Completed!")