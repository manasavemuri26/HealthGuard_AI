import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# optional: print accuracy
accuracy = model.score(X_test, y_test)
print("Training complete. Accuracy:", accuracy)

# Save model
joblib.dump(model, 'models/diabetes_model.pkl')
print("Model saved to models/diabetes_model.pkl")


