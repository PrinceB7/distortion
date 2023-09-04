import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv('data/combined_and_shuffled.csv')
X = df.drop('label', axis=1)
y = df['label']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model = xgb.XGBClassifier(
    objective='binary:logistic',  # Binary classification
    n_estimators=100,             # Number of trees (you can adjust this)
    max_depth=3,                  # Maximum depth of each tree (you can adjust this)
    random_state=42               # For reproducibility
)

model.fit(X_train, y_train)

# joblib.dump(model, 'weights/xgboost_model.pkl')

# model = joblib.load('weights/xgboost_model.pkl')

#evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
