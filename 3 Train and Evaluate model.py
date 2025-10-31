# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def predict_association(drug_name, disease_name) :
#     try :
#         d_enc = le_drug.transform([drug_name])[0]
#         dis_enc = le_disease.transform([disease_name])[0]
#         pred = model.predict([[d_enc, dis_enc]])[0]
#         print(f"Prediction for ({drug_name} <-> {disease_name}) : ",
#               f"{'Associated' if pred == 1 else 'Not Associated'}")
#     except Exception as e :
#         print("ERROR ! \n", e)

# # Loading the final dataset
# # df = pd.read_csv("C:/Users/Administrator/Desktop/ML Project/data/final_dataset.csv")
# df = pd.read_csv("C:/Users/Administrator/Desktop/ML Project/data/balanced_dataset.csv")

# X = df[['drug_encoded', 'disease_encoded']]
# y = df['label']

# # Splitting the dataset into TEST / TRAIN
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# # Training the Random Forest Classifier model
# model = RandomForestClassifier(n_estimators = 200, random_state = 42)
# model.fit(X_train, y_train)

# # Evaluating the model
# y_pred = model.predict(X_test)
# print("\n Model Evaluation Results :-- ")
# print(classification_report(y_test, y_pred))
# print("\nConfusion Matrix : \n", confusion_matrix(y_test, y_pred))


# # Plotting the results

# # Plot feature importance
# importances = model.feature_importances_
# features = ['drug_encoded', 'disease_encoded']

# plt.barh(features, importances)
# plt.xlabel("Importance Score")
# plt.title("Feature Importance - Random Forest Model")
# plt.show()


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load the balanced dataset
data_path = "C:/Users/Administrator/Desktop/ML Project/data"
df = pd.read_csv(f"{data_path}/balanced_dataset.csv")

print("Data loaded. Shape:", df.shape)
print(df.head())

# STEP 2: Encode categorical drug/disease IDs
le_drug = LabelEncoder()
le_disease = LabelEncoder()

df['drug_encoded'] = le_drug.fit_transform(df['drug_id'])
df['disease_encoded'] = le_disease.fit_transform(df['disease_id'])

# STEP 3: Prepare features & labels
X = df[['drug_encoded', 'disease_encoded']]
y = df['label']

# STEP 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# STEP 5: Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# STEP 6: Evaluate
y_pred = model.predict(X_test)

print("\nModel Evaluation Results:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional — Confusion Matrix Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
