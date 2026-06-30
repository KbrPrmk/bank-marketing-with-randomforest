<h1 align="center">🏦 Bank Marketing Prediction App</h1>
<p align="center"><b>A machine learning web application that predicts whether a customer will subscribe to a term deposit product using bank marketing campaign data.
</b></p>
<p align="center">
    <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
    <img src="https://img.shields.io/badge/Model-Random%20Forest-darkgreen" />
    <img src="https://img.shields.io/badge/Interface-Gradio-orange" />
    <img src="https://img.shields.io/badge/Deploy-Hugging%20Face-yellow" />
    <img src="https://img.shields.io/badge/Type-Classification-blueviolet">
    <img src="https://img.shields.io/badge/License-MIT-yellow">
</p>

---


 
# 🚀 Live Demo
 
```txt
https://huggingface.co/spaces/KubraParmak/bank-marketing-demo
```
 
---
 
# 📌 Project Overview
 
This project uses the Bank Marketing Dataset to predict customer subscription behavior for term deposit campaigns.
 
The model analyzes:
 
* Customer demographics
* Loan information
* Contact details
* Economic indicators
* Previous campaign results
and estimates the probability of subscription.
 
---
 
# 🎯 Objective
 
The goal is to classify customers into two categories:
 
* ✅ Likely to Subscribe
* ❌ Not Likely to Subscribe
This can help financial institutions improve campaign efficiency and reduce unnecessary calls.
 
---
 
# 📊 Dataset
 
Dataset Source:
 
* UCI Machine Learning Repository
* Bank Marketing Dataset
Target variable:
 
```python
y
```
 
Where:
 
* `yes` → customer subscribed
* `no` → customer did not subscribe
---
 
# ⚙️ Data Preprocessing & EDA
 
A detailed exploratory data analysis (EDA) and preprocessing pipeline was applied before model training.
 
## 📊 Exploratory Data Analysis (EDA)
 
The following analyses were performed:
 
* Missing value analysis
* Target variable distribution analysis
* Numerical feature analysis
* Categorical feature analysis
* Correlation analysis
* Pairplot analysis
These analyses helped identify feature distributions, class imbalance, and relationships between variables.
 
## 🧹 Data Cleaning
 
The dataset was cleaned and prepared before training. Steps included: checking missing values, handling categorical variables, removing inconsistencies, verifying feature types, preparing model-ready inputs.
 
## 🔠 One-Hot Encoding
 
Categorical variables were transformed into numerical representations using one-hot encoding (`pd.get_dummies`).
 
## ✨ Feature Engineering
 
* `was_contacted_before` — whether the customer had previous contact history
* `contact_intensity` — campaign intensity using current and previous contacts
* `economic_pressure` — derived from `euribor3m * emp.var.rate`
* `duration_cat` — call duration transformed into categorical groups
## ⚖️ Imbalanced Data Handling
 
Instead of using SMOTE, the project used `class_weight="balanced"` to preserve original data distribution while increasing sensitivity to minority classes.
 
## ✂️ Data Splitting
 
The dataset was split using stratified sampling (Training / Validation / Test) to preserve class distribution across all subsets.
 
---
 
# 🤖 Machine Learning Model
 
```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
```
 
---
 
# ⚠️ Important Note About Data Leakage (`duration`)
 
The `call duration` feature is, by a meaningful margin, the strongest predictor in this dataset (see Feature Importance below) — but it is also a **data leakage risk**: a call's duration is only known *after* the call has ended, so it cannot realistically be used at prediction time, before a call is placed.
 
This is a real limitation, not a footnote. **The headline metrics reported below (AUC 0.9496, Accuracy 0.89) include `duration` and `duration_cat` as features, and therefore overstate the model's real-world, pre-call predictive power.** A model meant for actual campaign targeting (i.e. "should we call this customer at all?") should be trained and evaluated **without** `duration`/`duration_cat`.
 
| Model variant | Status |
|---|---|
| With `duration` (reported below) | ✅ Trained — useful for understanding what drives subscription, not for pre-call targeting |
| Without `duration` (pre-call deployment model) | 🔲 Not yet trained — planned next step, see Roadmap |
 
If you're evaluating this project for deployment relevance rather than as an EDA/feature-engineering exercise, treat the metrics below as an upper bound, not as the deployable model's expected performance.
 
---
 
# 📈 Model Performance (includes `duration` — see leakage note above)
 
| Metric   | Score  |
| -------- | ------ |
| AUC-ROC  | 0.9496 |
| Accuracy | 0.89   |
| Recall   | 0.90   |
 
---
 
# 🔍 Feature Importance
 
1. duration
2. duration_cat
3. economic_pressure
4. euribor3m
5. nr.employed
Call duration was identified as the strongest predictor — which is precisely why it's flagged above as a leakage concern rather than presented as a clean result.
 
---
 
# 🗺️ Roadmap
 
- [ ] Retrain and report metrics for a **duration-free** model (the actually deployable version for pre-call targeting)
- [ ] Compare duration-free model performance against this duration-included baseline to quantify the real-world cost of removing the leaky feature
- [ ] Re-run feature importance on the duration-free model to identify which legitimate, pre-call features matter most
---
 
# 🖥️ Web Application
 
The application was deployed using Gradio and Hugging Face Spaces. Users can enter customer information through an interactive interface and receive real-time predictions.
 
> Note: the deployed demo currently uses the `duration`-included model described above. Per the leakage note, predictions from the live demo should be read as illustrative of the dataset's patterns rather than as a production-ready pre-call scoring tool.
 
---
 
# 📁 Project Structure
 
```bash
.
├── app.py
├── bank-additional-full.csv
├── model.pkl
├── columns.pkl
├── requirements.txt
├── README.md
├── train.ipynb
```
 
---
 
# ▶️ Run Locally
 
```bash
git clone https://huggingface.co/spaces/KubraParmak/bank-marketing-demo
pip install -r requirements.txt
python app.py
```
 
---
 
# 🧪 Example Prediction
 
Example scenario with high subscription probability (illustrative — uses the duration-included model):
 
| Feature           | Value   |
| ----------------- | ------- |
| Duration          | 850     |
| Previous Outcome  | success |
| Campaign Contacts | 1       |
| pdays             | 3       |
 
Expected prediction:
 
```txt
✅ Likely to Subscribe
```
 
---
 
# 🛠️ Technologies Used
 
* Python
* Pandas
* NumPy
* Scikit-learn
* Gradio
* Hugging Face
---
 
# 👩‍💻 Developer
 
Kübra Parmak
 
Machine Learning & Data Analysis Project
 


