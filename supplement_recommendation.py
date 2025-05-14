import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("classification_ready_supplement_dataset.csv")

# Encode 'Sex' as binary
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # Male=1, Female=0

# Define feature columns and target dose classes
target_cols = [col for col in df.columns if col.endswith('_dose_class')]
feature_cols = [col for col in df.columns if col not in target_cols + ['PatientID']]

# Prepare data
X = df[feature_cols]
y = df[target_cols]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MultiOutputClassifier with RandomForest
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_pred_df = pd.DataFrame(y_pred, columns=target_cols)

# Generate classification reports for each supplement
reports = {}
for col in target_cols:
    report = classification_report(y_test[col], y_pred_df[col], output_dict=True, zero_division=0)
    summary = {
        'accuracy': report['accuracy'],
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
        'f1_macro': report['macro avg']['f1-score']
    }
    reports[col] = summary

# Display the evaluation results
results_df = pd.DataFrame(reports).T.round(3)
print("\nSupplement Classification Evaluation Metrics:")
print(results_df)
