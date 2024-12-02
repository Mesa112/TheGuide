import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
file_path = "CAR/CTP_Model1.csv"
data = pd.read_csv(file_path, low_memory=False)

# Function to remove outliers using IQR
def remove_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers and unrealistic prices
data = remove_outliers_iqr(data, 'price', multiplier=2)
data = data[data['price'] > 100]

# Feature engineering
def create_features(df):
    df = df.copy()
    current_year = 2024
    df['age'] = current_year - df['year']
    df['age_squared'] = df['age'] ** 2
    df['mileage_per_year'] = np.clip(df['odometer'] / (df['age'] + 1), 0, 200000)
    return df

data = create_features(data)

# Handle categorical features
categorical_features = ['make', 'model', 'condition', 'fuel', 'title_status', 
                        'transmission', 'drive', 'size', 'type', 'paint_color']

label_encoders = {}
encoding_dict = {}  # To save mappings for the app

for feature in categorical_features:
    if feature in data.columns:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        label_encoders[feature] = le
        # Save mapping for later use
        encoding_dict[feature] = dict(zip(le.classes_, le.transform(le.classes_)))

# Save the encoding dictionary to a CSV
encoding_df = pd.DataFrame.from_dict(encoding_dict, orient='index').transpose()
encoding_df.to_csv("categorical_encodings.csv", index=False)

# Prepare features and labels
numeric_features = ['year', 'odometer', 'age', 'age_squared', 'mileage_per_year']
features = numeric_features + categorical_features
X = data[features]
y = np.log1p(data['price'])  # Log-transform the price for better model performance

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and regression
model = Pipeline([
    ('scaler', RobustScaler()),
    ('regressor', RandomForestRegressor(
        n_estimators=300, max_depth=25, random_state=42, n_jobs=-1))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

# Save the model and encoders
joblib.dump(model, "car_price_modelv3.pkl")
print("Model saved successfully.")

viz_path = '/Users/estebanm/Desktop/carShopping_tool/CAR/visualizations'
os.makedirs(viz_path, exist_ok=True)

# 1. Price Distribution Plot
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='price', bins=50)
plt.title('Price Distribution')
plt.savefig(os.path.join(viz_path, 'price_distribution_plot.png'))
plt.close()

# 2. Actual vs Predicted Plot
actual_prices = np.expm1(y_test)
predicted_prices = np.expm1(y_pred)

plt.figure(figsize=(10, 6))
plt.scatter(actual_prices, predicted_prices, alpha=0.5)
plt.plot([actual_prices.min(), actual_prices.max()], [actual_prices.min(), actual_prices.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.savefig(os.path.join(viz_path, 'actual_vs_predicted_scatter.png'))
plt.close()

# 3. Feature Importance Plot
feature_importance = model.named_steps['regressor'].feature_importances_
feature_names = numeric_features + categorical_features

plt.figure(figsize=(12, 6))
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=True)
plt.barh(importance_df['feature'], importance_df['importance'])
plt.title('Feature Importance')
plt.savefig(os.path.join(viz_path, 'feature_importance_plot.png'))
plt.close()

# 4. Residuals Distribution Plot
residuals = actual_prices - predicted_prices
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.savefig(os.path.join(viz_path, 'residuals_distribution_plot.png'))
plt.close()