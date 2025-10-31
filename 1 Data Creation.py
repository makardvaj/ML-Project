import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def clean_entity(x):
    if isinstance(x, str) :
        return x.split('::')[-1]
    else :
        return x

data_path = "C:/Users/Administrator/Desktop/ML Project/data"

# Reading the .tsv (tab-seperated values) file
df = pd.read_csv("C:/Users/Administrator/Desktop/ML Project/DRKG/drkg.tsv", sep = "\t")
df.columns = ['head', 'relation', 'tail']
# print(df.shape)
# print(df.head())

# Filtering "drug-disease" relations
drug_disease_df = df[df['relation'].str.contains('treats|associates', case = False, na = False)]
print(drug_disease_df.shape)
print(drug_disease_df.head())

# Cleaning Entity Names
drug_disease_df['drug_id'] = drug_disease_df['head'].apply(clean_entity)
drug_disease_df['disease_id'] = drug_disease_df['tail'].apply(clean_entity)

# Creating binary labels
# 1 = treats / associated with
# 2 = for future negative sampling
drug_disease_df['label'] = 1
drug_disease_df[['drug_id', 'disease_id', 'label']].to_csv((data_path + "/associations.csv"), index=False)
print("associations.csv saved succesfully !")

# Extracting lists of unique drugs and unique diseases
drugs = pd.DataFrame({'drug_id': drug_disease_df['drug_id'].unique()})
diseases = pd.DataFrame({'disease_id': drug_disease_df['disease_id'].unique()})

drugs.to_csv((data_path + "/drugs.csv"), index=False)
diseases.to_csv((data_path + "/diseases.csv"), index=False)

print(f"drugs.csv : ({len(drugs)} entries)")
print(f"diseases.csv : ({len(diseases)} entries)")

