import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

data_path = 'server_data_advanced.csv'
df = pd.read_csv(data_path)

print("First 5 Rows")
print(df.head())

print("Dataset Info")
print(df.info())

print("Summary Statistics")
print(df.describe())

print("Missing values")
print(df.isnull().sum())

encoders = {}
categorical_cols = ['Provider', 'Server_Type', 'Region', 'OS_Type']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("Encoded Data Sample")
print(df.head())

X = df.drop(columns=['CPU_Load'])
y = df['CPU_Load']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("----- Model Evaluation -----")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")
print(f"R2 Score: {r2_score(y_test, predictions) * 100:.2f}%")

if not os.path.exists("static"):
    os.makedirs("static")

plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=predictions, scatter_kws={'alpha':0.5, 'color':'#023e8a'}, line_kws={'color':'red'})
plt.xlabel("Actual CPU Load")
plt.ylabel("Predicted CPU Load")
plt.title("Actual vs Predicted Regression")
plt.savefig("static/regression_plot.png")
plt.close()

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_names[indices], y=importances[indices], palette="Blues_r")
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.close()

print("Saving Artifacts...")
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("Training Complete. Files saved.")