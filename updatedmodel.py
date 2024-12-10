import pandas as pd
import joblib

file_path = 'processed_modified.csv'
data = pd.read_csv(file_path)

'''category_mappings = {
    'gender': {0: 'Male', 1: 'Female'},
    'hypertension': {0: 'No', 1: 'Yes'},
    'heart_disease': {0: 'No', 1: 'Yes'},
    'smoking_status': {
        0: 'Never Smokes',
        1: 'Formerly Smokes',
        2: 'Smokes',
        3: 'Unknown'
    },
    'Residence_type': {0: 'Urban', 1: 'Rural'},
    'work_type': {
        0: 'Child',
        1: 'Never worked',
        2: 'Self-Employed',
        3: 'Private',
        4: 'Government employed'
    }
}

for column, mapping in category_mappings.items():
    if column in data.columns:
        data[column] = data[column].replace(mapping)
'''
print(data.head())
print(data.info())
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler  


X = data.drop(columns=['stroke']) 
y = data['stroke']  

scaler = StandardScaler()
numerical_columns = ['age', 'avg_glucose_level', 'bmi']
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
from sklearn.compose import ColumnTransformer

categorical_columns = ['gender', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease']
numerical_columns = ['age', 'avg_glucose_level', 'bmi']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', 'passthrough', categorical_columns)
    ]
)

X_transformed = preprocessor.fit_transform(X)

X = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf_model.fit(X_train_resampled, y_train_resampled)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("Classification Report:\n", classification_report(y_test, y_pred))
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

threshold = 0.3
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

print("Adjusted Classification Report:\n", classification_report(y_test, y_pred_adjusted))
joblib.dump(rf_model, 'model.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')