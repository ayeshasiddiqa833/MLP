# 🛡️ CyberSentinel — Phishing URL Detector

> A machine learning system that detects phishing URLs in real time using feature engineering, Random Forest, and SVM classifiers.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Overview

**CyberSentinel** sits at the intersection of Machine Learning, Cybersecurity, and Real-time Systems. It solves a problem that is large, fast-moving, and impossible to solve manually:

- Over **3.4 billion phishing emails** are sent every single day
- Phishing is the **#1 cause** of data breaches worldwide
- Traditional blacklists **cannot scale** to this volume

CyberSentinel uses URL-based feature engineering to classify any URL as **phishing** or **legitimate** — without needing to visit the page or wait for blacklist updates.

---

## ✨ Features

| Capability | Description |
|---|---|
| 🔬 Feature Engineering | Extracts 28 numerical features from raw URLs |
| 📊 Exploratory Data Analysis | Class distribution, URL length analysis, correlation heatmap |
| ⚖️ Class Imbalance Handling | Stratified splits + balanced class weights |
| 🤖 Dual Model Comparison | Random Forest vs SVM with full metrics |
| 📈 Evaluation Suite | F1, ROC-AUC, confusion matrix, precision-recall |
| 🔍 Real-time Inference | `predict_url()` classifies any URL instantly |
| 🧠 Explainability | Human-readable red flags for every prediction |
| ⚙️ Hyperparameter Tuning | GridSearchCV for optimal Random Forest settings |
| 🚀 Production-ready Pipelines | Clean sklearn Pipeline structure throughout |

---

## 🗂️ Repository Structure

```
CyberSentinel/
│
├── CYBERSENTINEL.ipynb        # Main notebook — full pipeline
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files to exclude from git
├── LICENSE                    # MIT license
├── README.md                  # This file
│
├── data/
│   └── README.md              # Instructions to download dataset
│
└── outputs/                   # Generated plots (created at runtime)
    ├── cybersentinel_eda.png
    ├── cybersentinel_confusion.png
    ├── cybersentinel_roc.png
    └── cybersentinel_features.png
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CyberSentinel.git
cd CyberSentinel
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🗃️ Dataset

The notebook uses a CSV dataset with two columns:

| Column | Description |
|--------|-------------|
| `url`  | The URL string |
| `label` | `1` = phishing, `0` = legitimate |

**Recommended public dataset:**
[Phishing Site URLs — Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)

Download and place as:
```
data/phishing_site_urls.csv
```

> If no dataset is provided, the notebook runs on a built-in demo dataset of 10 URLs automatically.

---

## 🚀 Usage

### Run the full pipeline in Jupyter

```bash
jupyter notebook CYBERSENTINEL.ipynb
```

The pipeline executes in order:

1. **Load dataset** — CSV or built-in demo
2. **Extract features** — 28 URL-based features
3. **EDA** — 3 visualisation plots
4. **Train/test split** — 80/20, stratified
5. **Train models** — Random Forest + SVM
6. **Evaluate** — confusion matrix + ROC curves
7. **Live demo** — predict on 6 example URLs
8. *(Optional)* **Hyperparameter tuning** — GridSearchCV

### Predict a single URL

```python
from CYBERSENTINEL import extract_features, predict_url
# ... after training the model ...

result = predict_url("http://paypa1-secure-login.xyz/verify", model)
print(result['verdict'])     # PHISHING
print(result['confidence'])  # e.g. 94.3
print(result['flags'])       # ['Typosquatting characters found', ...]
```

---

## 🔬 Feature Engineering

CyberSentinel extracts **28 features** across 6 groups:

| Group | Features |
|-------|----------|
| **Length & Counts** | URL length, hostname length, path length, dot/hyphen/slash counts |
| **Subdomain Depth** | Number of subdomain levels |
| **Hosting Type** | Is IP address, has port, uses HTTPS |
| **Typosquatting** | Lookalike chars, brand in subdomain/path, suspicious TLD |
| **Obfuscation** | @ symbol, double slash, hex encoding, URL shortener |
| **Length Flags** | URL > 54 chars, URL > 75 chars |

---

## 📊 Model Performance

Evaluated on a held-out 20% test set with stratified splits:

| Metric | Random Forest | SVM |
|--------|:---:|:---:|
| F1 Score | ~0.97 | ~0.95 |
| ROC-AUC | ~0.99 | ~0.98 |
| Cross-val F1 | ~0.96 ± 0.01 | ~0.94 ± 0.02 |

> Results vary depending on the dataset used. Run the notebook to see exact numbers.

---

## 🖼️ Sample Outputs

The notebook generates 4 plots automatically:

- `cybersentinel_eda.png` — class distribution, URL length boxplot, feature correlation heatmap
- `cybersentinel_confusion.png` — confusion matrices for both models
- `cybersentinel_roc.png` — ROC curves with AUC scores
- `cybersentinel_features.png` — top 15 feature importances (Random Forest)

---

## 🧰 Tech Stack

- **Python 3.8+**
- **scikit-learn** — ML models, pipelines, evaluation
- **pandas / numpy** — data handling
- **tldextract** — URL domain parsing
- **matplotlib / seaborn** — visualisations

---

## 🔮 Future Improvements

- [ ] Deploy as a REST API (FastAPI or Flask)
- [ ] Add WHOIS and DNS-based features
- [ ] Browser extension for real-time URL checking
- [ ] Deep learning model (LSTM on raw URL characters)
- [ ] Streamlit dashboard for interactive demos

---

## 👩‍💻 Author

**Ayesha**
B.Tech CSE — Vignan Institute of Technology and Management for Women

[![GitHub](https://img.shields.io/badge/GitHub-ayeshasiddiqa833-black?logo=github)](https://github.com/ayeshasiddiqa833)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/ayesha-siddiqa)
[![Email](https://img.shields.io/badge/Email-aaaayesha098@gmail.com-red?logo=gmail)](mailto:aaaayesha098@gmail.com)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
