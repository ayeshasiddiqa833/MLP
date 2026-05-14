# Dataset

The CyberSentinel notebook expects a CSV file with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `url`  | string | The full URL to classify |
| `label` | integer | `1` = phishing, `0` = legitimate |

## Recommended Dataset

**Phishing Site URLs** — available on Kaggle:

🔗 https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls

### Download steps

1. Create a free Kaggle account if you don't have one
2. Go to the link above and click **Download**
3. Extract and rename the file to `phishing_site_urls.csv`
4. Place it in this `data/` folder

```
data/
└── phishing_site_urls.csv   ← place file here
```

### Load it in the notebook

```python
df = load_dataset('data/phishing_site_urls.csv')
```

## Running without a dataset

If no CSV is provided, the notebook automatically uses a small built-in demo dataset of 10 URLs (3 legitimate, 7 phishing). This is enough to run the full pipeline and see all outputs, but the model accuracy metrics will not be meaningful.

```python
df = load_dataset()   # no argument = uses demo data
```

## Alternative Datasets

Other compatible phishing URL datasets:

- [UCI ML Repository — Phishing Websites](https://archive.ics.uci.edu/ml/datasets/phishing+websites)
- [PhishTank](https://phishtank.org/developer_info.php) — live phishing URL feed
- [OpenPhish](https://openphish.com/) — community phishing intelligence

Any dataset works as long as it has a `url` column and a `label` (or `status`/`result`) column.
