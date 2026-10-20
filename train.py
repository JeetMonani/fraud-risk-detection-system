import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from category_encoders import TargetEncoder

# 1. Load Data
data = pd.read_csv(r"C:\Users\user\Documents\Fraudulent_E-Commerce_Transaction_Data.csv", on_bad_lines='skip')

# 2. Feature Engineering: Address Mismatch
if 'Shipping Address' in data.columns and 'Billing Address' in data.columns:
    data['Address_Mismatch'] = (data['Shipping Address'] != data['Billing Address']).astype(int)

# 3. Separate Features (X) and Target (y)
y = data["Is Fraudulent"]

cols_to_drop = ['Transaction ID', 'Customer ID', 'Transaction Date', 'IP Address',
                'Shipping Address', 'Billing Address', 'Is Fraudulent']
cols_to_drop = [col for col in cols_to_drop if col in data.columns]
X_raw = data.drop(columns=cols_to_drop)

# 4. Memory-Safe Encoding (Target Encoding instead of One-Hot)
# Find all text/categorical columns remaining in your features
categorical_cols = X_raw.select_dtypes(include=['object', 'category', 'str']).columns.tolist()

print(f"Applying Target Encoding to columns: {categorical_cols}")
encoder = TargetEncoder(cols=categorical_cols)
X = encoder.fit_transform(X_raw, y)

# 5. Train/Test Split (Stratified to handle class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Scaling Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Model Training
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, class_weight='balanced', n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=300, random_state=42, class_weight='balanced')
}

for name, model in models.items():
    print(f"\nTraining {name} on E-commerce Data...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"\n{'#'*10} {name} Evaluation {'#'*10}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 8. Feature Importance (Random Forest)
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices], palette="mako")
plt.title("Top 10 Fraud Risk Indicators in E-Commerce", fontsize=14)
plt.xlabel("Predictive Weight")
plt.ylabel("E-commerce Features")
plt.tight_layout()
plt.show()

# 9. Save Artifacts for FastAPI & Streamlit
joblib.dump(rf_model, 'fraud_model.pkl')
joblib.dump(scaler, 'data_scaler.pkl')
joblib.dump(encoder, 'data_encoder.pkl') # Save the encoder so FastAPI can process live web traffic!
joblib.dump(X.columns.tolist(), 'model_features.pkl') 

print("\nModel training successful! Artifacts saved.")