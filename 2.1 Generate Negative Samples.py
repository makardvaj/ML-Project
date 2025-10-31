import pandas as pd
import numpy as np
from itertools import product

data_path = "C:/Users/Administrator/Desktop/ML Project/data"

# Load existing associations
pos = pd.read_csv(f"{data_path}/associations.csv")
pos['label'] = 1

# Load unique drugs & diseases
drugs = pd.read_csv(f"{data_path}/drugs.csv")['drug_id'].tolist()
diseases = pd.read_csv(f"{data_path}/diseases.csv")['disease_id'].tolist()

# All possible combinations
all_pairs = pd.DataFrame(list(product(drugs, diseases)), columns=['drug_id', 'disease_id'])

# Remove the positive (known) pairs
merged = all_pairs.merge(pos[['drug_id', 'disease_id']], on=['drug_id', 'disease_id'], how='left', indicator=True)
neg = merged[merged['_merge'] == 'left_only'][['drug_id', 'disease_id']]

# Sample equal number of negatives
neg = neg.sample(n=len(pos), random_state=42)
neg['label'] = 0

# Combine
balanced_df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
balanced_df.to_csv(f"{data_path}/balanced_dataset.csv", index=False)

print("Created balanced_dataset.csv with POSITIVE & NEGATIVE samples")
print(balanced_df['label'].value_counts())