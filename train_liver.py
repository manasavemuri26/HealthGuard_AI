import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv('liver.csv')

# 2. Some liver datasets have a 'Dataset' column as target or 'Result'
#   Check your dataset, here we assume target column is 'Dataset' or 'Result'

# Handle categorical 'Gender' field
if 'Gender' in df.columns:
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

# Here we assume the target column is 'Dataset' where 1 = disease, 2 = no disease
# You can change it to the actual name in your csv
target_col = 'Dataset'  # or change to 'Result' if your dataset has that
X = df.drop(target_col, axis=1)
y = df[target_col]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create model
model = RandomForestClassifier()

# 5. Fit model
model.fit(X_train, y_train)

# 6. Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Liver Model Accuracy:", acc)

# 7. Save model
joblib.dump(model, 'models/liver_model.pkl')
