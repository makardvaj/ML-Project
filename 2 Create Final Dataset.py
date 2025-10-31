from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Loading the drug-disease association data
df = pd.read_csv("C:/Users/Administrator/Desktop/ML Project/data/associations.csv")

# Encoding the IDs in a numeric format
le_drug = LabelEncoder()
le_disease = LabelEncoder()

df['drug_encoded'] = le_drug.fit_transform(df['drug_id'])
df['disease_encoded'] = le_disease.fit_transform(df['disease_id'])

# Selecting final feature columns
final_df = df[['drug_id', 'disease_id', 'drug_encoded', 'disease_encoded', 'label']]

# Saving it
final_df.to_csv("C:/Users/Administrator/Desktop/ML Project/data/final_dataset.csv", index=False)
print("final_dataset.csv created successfully !")
print(final_df.shape)
print(final_df.head())