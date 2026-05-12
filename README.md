
# 🏦 Bank Marketing Prediction App
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-darkgreen)
![Gradio](https://img.shields.io/badge/Interface-Gradio-orange)
![Hugging Face](https://img.shields.io/badge/Deploy-Hugging%20Face-yellow)
![Machine Learning](https://img.shields.io/badge/Type-Classification-blueviolet)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning web application that predicts whether a customer will subscribe to a term deposit product using bank marketing campaign data.

Built with:

* Python
* Scikit-learn
* Random Forest
* Gradio
* Hugging Face Spaces


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

---

## 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Missing value analysis
* Target variable distribution analysis
* Numerical feature analysis
* Categorical feature analysis
* Correlation analysis
* Pairplot analysis

These analyses helped identify feature distributions, class imbalance, and relationships between variables.

---

## 🧹 Data Cleaning

The dataset was cleaned and prepared before training.

Steps included:

* Checking missing values
* Handling categorical variables
* Removing inconsistencies
* Verifying feature types
* Preparing model-ready inputs

---

## 🔠 One-Hot Encoding

Categorical variables were transformed into numerical representations using one-hot encoding.

Example:

```python
pd.get_dummies(...)
```

---

## ✨ Feature Engineering

Several additional features were created to improve predictive performance.

### `was_contacted_before`

Indicates whether the customer had previous contact history.

### `contact_intensity`

Represents campaign intensity using current and previous contacts.

### `economic_pressure`

Generated from economic indicators:

```python
euribor3m * emp.var.rate
```

### `duration_cat`

Transforms call duration into categorical groups.

---

## ⚖️ Imbalanced Data Handling

The dataset contains class imbalance between subscription outcomes.

Instead of using SMOTE, the project used:

```python
class_weight="balanced"
```

This approach preserves original data distribution while increasing sensitivity to minority classes.

---

## ✂️ Data Splitting

The dataset was split using stratified sampling:

* Training Set
* Validation Set
* Test Set

This preserves class distribution across all subsets.

---

# 🤖 Machine Learning Model

Algorithm used:

```python
RandomForestClassifier
```

Parameters:

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

# 📈 Model Performance

| Metric   | Score  |
| -------- | ------ |
| AUC-ROC  | 0.9496 |
| Accuracy | 0.89   |
| Recall   | 0.90   |

---

# 🔍 Feature Importance

Top important features:

1. duration
2. duration_cat
3. economic_pressure
4. euribor3m
5. nr.employed

Call duration was identified as the strongest predictor.

---

# ⚠️ Important Note About Data Leakage

The `duration` variable may introduce potential data leakage because call duration becomes fully known only after the conversation ends.

This limitation should be considered when interpreting model performance.

---

# 🖥️ Web Application

The application was deployed using:

* Gradio
* Hugging Face Spaces

Users can enter customer information through an interactive interface and receive real-time predictions.

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

Clone repository:

```bash
git clone https://huggingface.co/spaces/KubraParmak/bank-marketing-demo
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run app:

```bash
python app.py
```

---

# 🧪 Example Prediction

Example scenario with high subscription probability:

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


