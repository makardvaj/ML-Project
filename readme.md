# Drug Repurposing Prediction using Machine Learning

## 🎯 Overview
This project aims to **predict potential new uses for existing drugs** — a process known as *drug repurposing* — using machine learning.  
We use the **Random Forest Classifier** to identify possible associations between drugs and diseases based on relationship data from biomedical knowledge graphs.

---

## ⚙️ Tech Stack
- **Language:** Python 3.11+
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib, joblib
- **Model Used:** Random Forest Classifier

---

## 📂 Project Structure
- `data/` – contains raw DRKG dataset and processed intermediate files.  
- `models/` – stores trained models and encoders for reuse.  
- `notebooks/` – contains modular scripts for each phase.  
- `main/` – includes the final integrated script and quick test file.  
- `results/` – visual outputs and performance reports.

---

## 🚀 How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
