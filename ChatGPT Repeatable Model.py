# ============================================================
# 4_Model_Save_and_Predict.py
# ============================================================
# This script saves your trained model and encoders,
# loads them later, and allows live prediction of drug–disease pairs.
# ============================================================

import joblib

# ============================================================
# Step 1: Save trained model and encoders
# ============================================================

# Assuming you have the following already trained/defined in memory:
# model, drug_encoder, disease_encoder

joblib.dump(model, 'drug_disease_model.pkl')
joblib.dump(drug_encoder, 'drug_encoder.pkl')
joblib.dump(disease_encoder, 'disease_encoder.pkl')

print("✅ Model and encoders saved successfully!")


# ============================================================
# Step 2: Load model and encoders later
# ============================================================

model = joblib.load('drug_disease_model.pkl')
drug_encoder = joblib.load('drug_encoder.pkl')
disease_encoder = joblib.load('disease_encoder.pkl')

print("✅ Model and encoders loaded successfully!")


# ============================================================
# Step 3: Live prediction function
# ============================================================

def predict_association(drug_id, disease_id):
    """
    Predicts whether a given drug–disease pair is likely associated.
    Prints both the prediction and model confidence.
    """
    try:
        # Encode input
        drug_encoded = drug_encoder.transform([drug_id])[0]
        disease_encoded = disease_encoder.transform([disease_id])[0]

        # Predict association
        pred = model.predict([[drug_encoded, disease_encoded]])[0]
        prob = model.predict_proba([[drug_encoded, disease_encoded]])[0][1]

        if pred == 1:
            print(f"✅ Predicted: {drug_id} is likely associated with {disease_id}")
        else:
            print(f"❌ Predicted: No known association between {drug_id} and {disease_id}")

        print(f"Confidence: {prob:.2f}")

    except Exception as e:
        print(f"⚠️ Error: {e}")


# ============================================================
# Step 4: Example predictions
# ============================================================

# Try predicting some known or new pairs
predict_association("DB00316", "MESH:D003924")
predict_association("DB01050", "MESH:D004194")


# ============================================================
# Step 5: Optional – View some valid IDs
# ============================================================

print("\n🔍 Example valid Drug IDs:")
print(list(drug_encoder.classes_)[:10])

print("\n🔍 Example valid Disease IDs:")
print(list(disease_encoder.classes_)[:10])

# ============================================================
# End of Script
# ============================================================
