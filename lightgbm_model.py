import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # or import pickle if you prefer pickle

class LightGBM:
    
    def __init__(self, model_name='lgbm_model.pkl'):
        self.params = {'learning_rate': 0.01,
          'num_leaves': 11, 
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'metric': 'f1', 
          'feature_fraction': 0.9, 
          'bagging_fraction': 0.9, 
          'bagging_freq': 5, 
          'seed': 42}
        self.model_name = model_name
        
    def train(self, train_ds, val_ds,
              num_boost_round=1000):
        self.model = lgb.train(self.params,
                               train_set=train_ds,
                               valid_sets=[val_ds],
                               num_boost_round=num_boost_round)
        self.save_model()
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred
    
    def save_model(self):
        joblib.dump(self.model, "weights/"+self.model_name)
        # Or you can use pickle:
        # with open(self.model_name, 'wb') as model_file:
        #     pickle.dump(self.model, model_file)

    def load_model(self):
        self.model = joblib.load(self.model_name)
        # Or you can use pickle:
        # with open(self.model_name, 'rb') as model_file:
        #     self.model = pickle.load(model_file)

df = pd.read_csv('data/combined_and_shuffled.csv')
X = df.drop('label', axis=1)
y = df['label']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

lgbm_model = LightGBM(model_name='lgbm_model.pkl')

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
lgbm_model.train(train_data, val_data)

y_pred = lgbm_model.predict(X_test)
accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
report = classification_report(y_test, (y_pred > 0.5).astype(int))

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
