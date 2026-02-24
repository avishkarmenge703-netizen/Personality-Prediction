import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. Load the dataset (update the filename if needed)
# ------------------------------------------------------------
data = pd.read_csv('personality_synthetic_dataset (1).csv')

# ------------------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------------------
X = data.drop('personality_type', axis=1)
y = data['personality_type']

# ------------------------------------------------------------
# 3. Encode the target labels (Extrovert, Introvert, Ambivert)
# ------------------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ------------------------------------------------------------
# 4. Train / test split (80% train, 20% test)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ------------------------------------------------------------
# 5. Feature scaling (optional but kept for consistency)
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 6. Train a baseline Random Forest model
# ------------------------------------------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# 7. Evaluate on the test set
# ------------------------------------------------------------
y_pred = rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline Random Forest Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ------------------------------------------------------------
# 8. Save the model, scaler, and label encoder for the web app
# ------------------------------------------------------------
joblib.dump(rf, 'personality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\n✅ Model, scaler, and label encoder saved successfully!")
print("   Files: personality_model.pkl, scaler.pkl, label_encoder.pkl")
