# Part 1: Import Libraries and Load Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your adjusted synthetic dataset
realistic_df = pd.read_csv("realistic_adjusted_synthetic_dataset.csv")

# Convert 'Sex' to binary
realistic_df['Sex'] = realistic_df['Sex'].map({'Male': 1, 'Female': 0})


# Part 2: Generate Boosted Supplement Labels with Probabilistic Logic
def boosted_dose_logic(row):
    logic = {}

    # Vitamin D3 (based on Vitamin D levels)
    if row['Vitamin_D'] < 30:
        logic['Vitamin_D3_dose_class'] = np.random.choice([3, 2], p=[0.75, 0.25])
    elif row['Vitamin_D'] < 50:
        logic['Vitamin_D3_dose_class'] = np.random.choice([2, 1], p=[0.7, 0.3])
    else:
        logic['Vitamin_D3_dose_class'] = 1

    # Folate (based on MCH)
    logic['Folate_dose_class'] = np.random.choice([3, 2, 1], p=[0.6, 0.3, 0.1]) if row['MCH'] < 27 else 2

    # Niacin (based on LDL)
    logic['Niacin_dose_class'] = np.random.choice([3, 2, 1], p=[0.6, 0.3, 0.1]) if row['LDL'] > 4 else np.random.choice([2, 1], p=[0.6, 0.4])

    # Zinc (based on Testosterone)
    logic['Zinc_dose_class'] = np.random.choice([3, 2], p=[0.7, 0.3]) if row['Testosterone'] < 12 else 1

    # Selenium (based on hsCRP and Cortisol)
    logic['Selenium_dose_class'] = 3 if row['hsCRP'] > 3 or row['Cortisol'] < 150 else 2

    return pd.Series(logic)

# Apply to dataset
boosted_targets = realistic_df.apply(boosted_dose_logic, axis=1)


# Part 3: Prepare Features and Train Model
# Define features and target
features = ['Age', 'Sex', 'Vitamin_D', 'Ferritin', 'LDL', 'Testosterone', 'hsCRP', 'Cortisol', 'MCH']
X = realistic_df[features]
y = boosted_targets

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train_scaled, y_train)


# Part 4: Predict and Evaluate
# Predict
y_pred = model.predict(X_test_scaled)
y_pred_df = pd.DataFrame(y_pred, columns=y.columns)

# Evaluate
for col in y.columns:
    print(f"\n===== {col} =====")
    print(classification_report(y_test[col], y_pred_df[col], zero_division=0))
