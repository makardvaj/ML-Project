# ============================================================
# Drug_Repurposing_Final_Model.py
# ============================================================
# A complete Python script for Drug Repurposing prediction
# using Random Forest and machine learning.
# Features:
#   - Load & preprocess data
#   - Train & evaluate model
#   - Save/load model and encoders
#   - Live predictions
#   - Visual analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib


# ============================================================
# Step 1: Load the dataset
# ============================================================

# Replace this path with your final dataset CSV
df = pd.read_csv("final_drug_disease_dataset.csv")

print(f"Data loaded. Shape: {df.shape}")
print(df.head())


# ============================================================
# Step 2: Encode categorical columns
# ============================================================

drug_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

df['drug_encoded'] = drug_encoder.fit_transform(df['drug_id'])
df['disease_encoded'] = disease_encoder.fit_transform(df['disease_id'])

# Keep relevant columns
X = df[['drug_encoded', 'disease_encoded']]
y = df['label']


# ============================================================
# Step 3: Split into train/test
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")


# ============================================================
# Step 4: Train Random Forest model
# ============================================================

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight='balanced'
)
model.fit(X_train, y_train)

print("Model training complete!")


# ============================================================
# Step 5: Evaluate model
# ============================================================

y_pred = model.predict(X_test)
print("\nModel Evaluation Results:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# Step 6: Feature Importance
# ============================================================

plt.figure(figsize=(5, 4))
importance = model.feature_importances_
plt.bar(['Drug Encoded', 'Disease Encoded'], importance, color='teal')
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.show()


# ============================================================
# Step 7: Save model and encoders
# ============================================================

joblib.dump(model, 'drug_disease_model.pkl')
joblib.dump(drug_encoder, 'drug_encoder.pkl')
joblib.dump(disease_encoder, 'disease_encoder.pkl')

print("Model and encoders saved successfully!")


# ============================================================
# Step 8: Load them later (simulate reuse)
# ============================================================

model = joblib.load('drug_disease_model.pkl')
drug_encoder = joblib.load('drug_encoder.pkl')
disease_encoder = joblib.load('disease_encoder.pkl')

print("Model and encoders loaded successfully!")


# ============================================================
# Step 9: Live Prediction Function
# ============================================================

def predict_association(drug_id, disease_id):
    """
    Predicts whether a given drug–disease pair is likely associated.
    Prints both the prediction and model confidence.
    """
    try:
        drug_encoded = drug_encoder.transform([drug_id])[0]
        disease_encoded = disease_encoder.transform([disease_id])[0]

        pred = model.predict([[drug_encoded, disease_encoded]])[0]
        prob = model.predict_proba([[drug_encoded, disease_encoded]])[0][1]

        if pred == 1:
            print(f"Predicted: {drug_id} is likely associated with {disease_id}")
        else:
            print(f"Predicted: No known association between {drug_id} and {disease_id}")

        print(f"Confidence: {prob:.2f}")

    except Exception as e:
        print(f"Error: {e}")


# ============================================================
# Step 10: Example Predictions
# ============================================================

print("\n🔮 Example Predictions:")
predict_association("DB00316", "MESH:D003924")
predict_association("DB01050", "MESH:D004194")


# ============================================================
# Step 11: Optional – View some valid IDs
# ============================================================

print("\n🔍 Example valid Drug IDs:")
print(list(drug_encoder.classes_)[:10])

print("\n🔍 Example valid Disease IDs:")
print(list(disease_encoder.classes_)[:10])

# ============================================================
# END OF SCRIPT
# ============================================================

