# ============================================================
#  CyberSentinel — Phishing URL Classifier
#  Google Colab-ready | Python 3 | scikit-learn
# ============================================================
# SETUP: Run this first in Colab
# !pip install tldextract pandas scikit-learn seaborn matplotlib
# ============================================================

import re
import socket
import pandas as pd
import numpy as np
import tldextract
import seaborn as sns
import matplotlib.pyplot as plt

from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from sklearn.pipeline import Pipeline


# ============================================================
# SECTION 1 — FEATURE EXTRACTION
# ============================================================

SUSPICIOUS_TLDS = {'.xyz', '.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.click', '.loan'}

BRAND_KEYWORDS = [
    'paypal', 'google', 'apple', 'amazon', 'microsoft', 'netflix',
    'facebook', 'instagram', 'twitter', 'linkedin', 'ebay', 'bank',
    'secure', 'login', 'signin', 'verify', 'account', 'update'
]

# Typosquatting substitution patterns (char → legitimate char)
TYPO_PATTERNS = {
    '0': 'o', '1': 'l', '3': 'e', '4': 'a',
    '5': 's', '6': 'g', '7': 't', '8': 'b',
    '@': 'a', '$': 's', '!': 'i'
}


def is_ip_address(hostname: str) -> int:
    """Returns 1 if hostname is a raw IPv4/IPv6 address, 0 otherwise."""
    try:
        socket.inet_aton(hostname)
        return 1
    except socket.error:
        pass
    # IPv6 rough check
    if re.match(r'^[\da-fA-F:]+$', hostname) and ':' in hostname:
        return 1
    return 0


def count_typosquatting_chars(domain: str) -> int:
    """Count suspicious character substitutions that mimic letters."""
    count = 0
    for char in domain:
        if char in TYPO_PATTERNS:
            count += 1
    return count


def has_brand_in_subdomain(ext) -> int:
    """Checks if a brand keyword appears in the subdomain (not main domain)."""
    subdomain = ext.subdomain.lower()
    for brand in BRAND_KEYWORDS:
        if brand in subdomain:
            return 1
    return 0


def has_suspicious_tld(ext) -> int:
    suffix = '.' + ext.suffix.lower() if ext.suffix else ''
    return 1 if suffix in SUSPICIOUS_TLDS else 0


def extract_features(url: str) -> dict:
    """
    Extract all heuristic features from a single URL string.
    Returns a flat dictionary of numeric features.
    """
    features = {}

    # --- Parse URL ---
    try:
        parsed = urlparse(url if url.startswith('http') else 'http://' + url)
        ext = tldextract.extract(url)
    except Exception:
        # Return zeroed features on parse failure
        return {k: 0 for k in _feature_names()}

    hostname  = parsed.netloc or ''
    path      = parsed.path or ''
    query     = parsed.query or ''
    domain    = ext.domain.lower()
    subdomain = ext.subdomain.lower()

    # ---- Structural Heuristics ----
    features['url_length']          = len(url)
    features['hostname_length']     = len(hostname)
    features['path_length']         = len(path)
    features['num_dots']            = url.count('.')
    features['num_hyphens']         = url.count('-')
    features['num_underscores']     = url.count('_')
    features['num_slashes']         = url.count('/')
    features['num_question_marks']  = url.count('?')
    features['num_ampersands']      = url.count('&')
    features['num_equals']          = url.count('=')
    features['num_at_symbols']      = url.count('@')          # red flag
    features['num_percent']         = url.count('%')          # encoding obfuscation
    features['num_digits_in_url']   = sum(c.isdigit() for c in url)

    # ---- Subdomain depth ----
    subdomain_parts = [s for s in subdomain.split('.') if s]
    features['subdomain_depth']     = len(subdomain_parts)   # ≥3 is suspicious

    # ---- Abnormal Features ----
    features['is_ip_address']       = is_ip_address(hostname.split(':')[0])
    features['has_port']            = 1 if ':' in hostname and not hostname.endswith(':443') else 0
    features['uses_https']          = 1 if parsed.scheme == 'https' else 0

    # ---- Identity / Typosquatting Heuristics ----
    features['typo_char_count']     = count_typosquatting_chars(domain)
    features['has_brand_in_subdomain'] = has_brand_in_subdomain(ext)
    features['brand_keyword_in_path'] = int(any(b in path.lower() for b in BRAND_KEYWORDS))
    features['suspicious_tld']      = has_suspicious_tld(ext)
    features['domain_has_numbers']  = int(bool(re.search(r'\d', domain)))

    # ---- Obfuscation signals ----
    features['has_at_symbol']       = 1 if '@' in url else 0
    features['has_double_slash']    = 1 if '//' in path else 0
    features['has_hex_encoding']    = 1 if '%' in url else 0
    features['url_shortener']       = int(domain in {
        'bit', 'tinyurl', 'goo', 't', 'ow', 'is', 'buff', 'short'
    })

    # ---- Length-based thresholds (binary) ----
    features['url_len_gt_54']       = 1 if len(url) > 54 else 0
    features['url_len_gt_75']       = 1 if len(url) > 75 else 0

    return features


def _feature_names():
    """Return the ordered list of feature names (for zero-padding on error)."""
    return list(extract_features('http://example.com').keys())


def build_feature_matrix(urls: list) -> pd.DataFrame:
    """
    Given a list of URLs, return a DataFrame of extracted features.
    """
    records = [extract_features(url) for url in urls]
    return pd.DataFrame(records)


# ============================================================
# SECTION 2 — LOAD & PREPARE DATASET
# ============================================================
# Option A: PhishTank CSV (download from https://phishtank.org/developer_info.php)
# Option B: UCI Phishing Dataset (from UCI ML Repository)
# Option C: Use the synthetic demo below for testing

def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Load and prepare the dataset.
    Expects columns: 'url' and 'label' (1=phishing, 0=legitimate).

    If no path given, a small synthetic demo set is returned.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        # Normalize label column name if needed
        if 'result' in df.columns:
            df = df.rename(columns={'result': 'label'})
        if 'status' in df.columns:
            df['label'] = df['status'].map({'phishing': 1, 'legitimate': 0})
        df = df[['url', 'label']].dropna()
        df['label'] = df['label'].astype(int)
    else:
        print("[INFO] No CSV path provided — using built-in demo dataset.")
        demo_data = {
            'url': [
                'https://www.google.com/search?q=test',
                'https://github.com/user/repo',
                'https://www.amazon.com/dp/B08N5WRWNW',
                'http://192.168.1.1/paypal/login/secure',
                'http://secure.g00gle.com.evil-domain.ru@phish.biz/login',
                'http://paypa1.com-secure.update-account.xyz/verify',
                'https://bit.ly/3xR9fake',
                'http://login.faceb00k.com/account/suspended',
                'https://www.wikipedia.org/wiki/Phishing',
                'http://193.203.1.45/ebay/update/signin',
            ],
            'label': [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
        }
        df = pd.DataFrame(demo_data)

    print(f"[INFO] Dataset loaded: {len(df)} records | "
          f"Phishing: {df['label'].sum()} | Legit: {(df['label']==0).sum()}")
    return df


# ============================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def run_eda(df: pd.DataFrame, features_df: pd.DataFrame):
    """
    Generate EDA plots:
    1. Class distribution
    2. URL length distribution by class
    3. Feature correlation heatmap
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('CyberSentinel — EDA', fontsize=14, fontweight='bold')

    # 1. Class distribution
    class_counts = df['label'].value_counts()
    axes[0].bar(
        ['Legitimate', 'Phishing'],
        [class_counts.get(0, 0), class_counts.get(1, 0)],
        color=['#1D9E75', '#E24B4A'], edgecolor='white', linewidth=0.5
    )
    axes[0].set_title('Class Distribution')
    axes[0].set_ylabel('Count')

    # 2. URL length by class
    combined = features_df.copy()
    combined['label'] = df['label'].values
    sns.boxplot(
        data=combined, x='label', y='url_length', ax=axes[1],
        palette={0: '#1D9E75', 1: '#E24B4A'}
    )
    axes[1].set_title('URL Length by Class')
    axes[1].set_xticklabels(['Legitimate', 'Phishing'])
    axes[1].set_xlabel('')

    # 3. Correlation heatmap (top features)
    top_features = [
        'url_length', 'num_dots', 'num_at_symbols', 'is_ip_address',
        'subdomain_depth', 'typo_char_count', 'uses_https',
        'suspicious_tld', 'brand_keyword_in_path', 'url_len_gt_75'
    ]
    corr_data = combined[top_features + ['label']].corr()
    sns.heatmap(
        corr_data[['label']].drop('label').sort_values('label', ascending=False),
        annot=True, fmt='.2f', cmap='RdYlGn', center=0,
        ax=axes[2], cbar=True, linewidths=0.5
    )
    axes[2].set_title('Feature Correlation with Label')
    axes[2].set_ylabel('')

    plt.tight_layout()
    plt.savefig('cybersentinel_eda.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[INFO] EDA saved to cybersentinel_eda.png")


# ============================================================
# SECTION 4 — MODEL TRAINING
# ============================================================

def train_models(X_train, X_test, y_train, y_test):
    """
    Train Random Forest and SVM classifiers.
    Returns dict of fitted models and their evaluation results.
    """
    results = {}

    # ---- Random Forest ----
    print("\n[1/2] Training Random Forest ...")
    rf_pipeline = Pipeline([
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight='balanced',   # handles class imbalance
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_pred  = rf_pipeline.predict(X_test)
    rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

    results['Random Forest'] = {
        'model':     rf_pipeline,
        'pred':      rf_pred,
        'proba':     rf_proba,
        'f1':        f1_score(y_test, rf_pred),
        'roc_auc':   roc_auc_score(y_test, rf_proba),
        'cv_scores': cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring='f1')
    }

    # ---- SVM ----
    print("[2/2] Training SVM ...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),   # SVM needs feature scaling
        ('clf', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,           # needed for ROC AUC
            random_state=42
        ))
    ])
    svm_pipeline.fit(X_train, y_train)
    svm_pred  = svm_pipeline.predict(X_test)
    svm_proba = svm_pipeline.predict_proba(X_test)[:, 1]

    results['SVM'] = {
        'model':     svm_pipeline,
        'pred':      svm_pred,
        'proba':     svm_proba,
        'f1':        f1_score(y_test, svm_pred),
        'roc_auc':   roc_auc_score(y_test, svm_proba),
        'cv_scores': cross_val_score(svm_pipeline, X_train, y_train, cv=5, scoring='f1')
    }

    return results


# ============================================================
# SECTION 5 — EVALUATION & VISUALISATION
# ============================================================

def evaluate_models(results: dict, y_test):
    """
    Print classification reports and plot:
    1. Confusion matrices (side by side)
    2. ROC curves (overlaid)
    3. Feature importances (RF only)
    """
    # --- Print reports ---
    for name, res in results.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(classification_report(y_test, res['pred'],
                                     target_names=['Legitimate', 'Phishing']))
        print(f"  ROC-AUC : {res['roc_auc']:.4f}")
        print(f"  CV F1   : {res['cv_scores'].mean():.4f} ± {res['cv_scores'].std():.4f}")

    # --- Confusion matrices ---
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res['pred'])
        sns.heatmap(
            cm, annot=True, fmt='d', ax=ax,
            cmap='Blues', linewidths=0.5,
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing']
        )
        ax.set_title(f'{name} — Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('cybersentinel_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- ROC Curves ---
    plt.figure(figsize=(7, 5))
    colors = ['#E24B4A', '#534AB7']
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['proba'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
                 color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CyberSentinel — ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cybersentinel_roc.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Feature Importance (Random Forest) ---
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model'].named_steps['clf']
        importances = pd.Series(
            rf_model.feature_importances_,
            index=_feature_names()
        ).sort_values(ascending=False).head(15)

        plt.figure(figsize=(8, 5))
        importances.plot(kind='barh', color='#534AB7', edgecolor='white')
        plt.gca().invert_yaxis()
        plt.title('Top 15 Feature Importances (Random Forest)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('cybersentinel_features.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\n[INFO] Top 5 features:")
        print(importances.head().to_string())


# ============================================================
# SECTION 6 — REAL-TIME PREDICTION INTERFACE
# ============================================================

def predict_url(url: str, model, threshold: float = 0.5) -> dict:
    """
    Classify a single URL as phishing or legitimate.

    Args:
        url       : The URL string to classify
        model     : A fitted sklearn Pipeline
        threshold : Probability cutoff (default 0.5; lower = stricter)

    Returns:
        dict with verdict, confidence, and triggered flags
    """
    features = extract_features(url)
    X = pd.DataFrame([features])
    prob_phish = model.predict_proba(X)[0][1]
    verdict    = 'PHISHING' if prob_phish >= threshold else 'LEGITIMATE'

    # Collect human-readable red flags
    flags = []
    if features['num_at_symbols'] > 0:        flags.append('@ symbol detected')
    if features['is_ip_address']:              flags.append('Raw IP address as host')
    if features['subdomain_depth'] >= 3:       flags.append(f'{features["subdomain_depth"]}-level subdomain depth')
    if features['typo_char_count'] > 0:        flags.append('Typosquatting characters found')
    if features['suspicious_tld']:             flags.append('Suspicious TLD')
    if features['url_len_gt_75']:              flags.append('URL length > 75 chars')
    if features['has_brand_in_subdomain']:     flags.append('Brand name in subdomain')
    if features['brand_keyword_in_path']:      flags.append('Brand keyword in path')
    if not features['uses_https']:             flags.append('No HTTPS')

    return {
        'url':        url,
        'verdict':    verdict,
        'confidence': round(prob_phish * 100, 1),
        'flags':      flags if flags else ['No suspicious flags found']
    }


def demo_predictions(model):
    """Run predictions on a set of example URLs and print results."""
    test_urls = [
        'https://www.google.com/search?q=machine+learning',
        'http://192.168.10.5/paypal/verify/account',
        'http://secure.g00gle.com.phish.ru@evil.biz/login',
        'https://github.com/openai/gpt-4',
        'http://paypa1-secure-login.xyz/update/credentials',
        'https://www.amazon.co.uk/dp/B0BTDVMZ9H',
    ]

    print("\n" + "="*60)
    print("  CyberSentinel — Live Prediction Demo")
    print("="*60)

    for url in test_urls:
        result = predict_url(url, model)
        icon = "🔴" if result['verdict'] == 'PHISHING' else "🟢"
        print(f"\n{icon} [{result['verdict']}] {result['confidence']}% confidence")
        print(f"   URL  : {url[:70]}{'...' if len(url) > 70 else ''}")
        for flag in result['flags']:
            print(f"   ⚑    {flag}")


# ============================================================
# SECTION 7 — HYPERPARAMETER TUNING (Optional / slow)
# ============================================================

def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Grid search over key RF hyperparameters.
    Run this only if you have time — can take 10–20 mins on large datasets.
    """
    param_grid = {
        'clf__n_estimators':     [100, 200, 300],
        'clf__max_depth':        [None, 20, 40],
        'clf__min_samples_split':[2, 5, 10],
        'clf__max_features':     ['sqrt', 'log2'],
    }
    pipeline = Pipeline([
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"[TUNING] Best params : {grid.best_params_}")
    print(f"[TUNING] Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ============================================================
# MAIN — Run full pipeline
# ============================================================

if __name__ == '__main__':

    # 1. Load data
    # For your real dataset: df = load_dataset('phishtank_urls.csv')
    df = load_dataset()

    # 2. Feature extraction
    print("\n[INFO] Extracting features ...")
    features_df = build_feature_matrix(df['url'].tolist())
    print(f"[INFO] Feature matrix shape: {features_df.shape}")
    print(features_df.describe().loc[['mean', 'std']].T.head(10).to_string())

    # 3. EDA
    run_eda(df, features_df)

    # 4. Train/test split
    X = features_df.values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # 5. Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # 6. Evaluate
    evaluate_models(results, y_test)

    # 7. Live demo
    best_model = results['Random Forest']['model']
    demo_predictions(best_model)

    # 8. (Optional) Hyperparameter tuning
    # best_model = tune_random_forest(X_train, y_train)

    print("\n[DONE] CyberSentinel pipeline complete.")
