import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load dataset (corrected separator to default comma)
df = pd.read_csv('heart.csv')  # Removed sep='\t'

# Optional: Print columns for debugging
print("Columns in dataset:", df.columns)

# 2. Set target column
target_col = 'target'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in data columns: {df.columns.tolist()}")

X = df.drop(target_col, axis=1)
y = df[target_col]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Heart Model Accuracy:", acc)

# 6. Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/heart_model.pkl')
print("Heart model saved successfully!")
