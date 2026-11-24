# 🎧 Spotify Hit Song Predictor

A machine learning project that predicts whether a song will be a **Hit** or **Not a Hit** using only its Spotify audio features. The best model is served through an interactive web interface built with Gradio.

---

## 📌 Overview

- **Task:** Binary classification (`target = 1` → Hit, `target = 0` → Not a Hit)
- **Data:** 6,398 songs from the 2010s, 15 audio features each
- **Models compared:** Decision Tree, Random Forest, SVM, XGBoost, Bagging
- **Best model:** Random Forest (**86.1%** accuracy)
- **Deployment:** Gradio web app with a shareable public link

---

## 📊 Dataset

File: `spotify_hits_10s.csv` (6,398 rows × 19 columns, no missing values).

The classes are perfectly balanced: **3,199 Hit** and **3,199 Not a Hit**, so accuracy is a fair metric.

### Dropped columns

| Column | Reason |
|---|---|
| `track` | Song name, text identifier and not an audio feature |
| `artist` | Artist name, 3,355 unique values and not an audio feature |
| `uri` | Spotify ID, nearly unique per row, would cause memorization |

### Features used (15)

| Feature | Meaning | Range |
|---|---|---|
| `danceability` | How suitable the track is for dancing | 0–1 |
| `energy` | Intensity and activity | 0–1 |
| `key` | Musical key (pitch class) | 0–11 |
| `loudness` | Overall loudness in dB | about -46 to 0 |
| `mode` | 0 = minor, 1 = major | 0 / 1 |
| `speechiness` | Amount of spoken words | 0–1 |
| `acousticness` | Likelihood the track is acoustic | 0–1 |
| `instrumentalness` | Likelihood the track has no vocals | 0–1 |
| `liveness` | Likelihood of a live audience | 0–1 |
| `valence` | Musical positivity | 0–1 |
| `tempo` | Speed in BPM | about 39–211 |
| `duration_ms` | Track length in milliseconds | 30 s – 29 min |
| `time_signature` | Beats per bar | 0–5 |
| `chorus_hit` | Time the first chorus starts, in seconds | 0–213 |
| `sections` | Number of sections in the track | 2–88 |

### Most related to the target

| Feature | Correlation with `target` |
|---|---|
| `instrumentalness` | -0.50 |
| `danceability` | +0.38 |
| `loudness` | +0.33 |
| `valence` | +0.20 |
| `duration_ms` | -0.19 |
| `acousticness` | -0.18 |

Hit songs tend to have vocals, be easier to dance to, be louder, be more positive and be a bit shorter.

---

## 🔧 Pipeline

### 1. Exploratory data analysis
- `describe()`, `info()` and missing-value check
- Target distribution (countplot)
- Correlation heatmap
- KDE and boxplots of every feature by target
- Energy vs danceability scatter plot
- Pairplot of `danceability`, `energy`, `valence`, `loudness`

### 2. Preprocessing

| Step | Columns | Purpose |
|---|---|---|
| `np.log1p` | `speechiness`, `acousticness`, `liveness` | Reduce skewness |
| IQR capping (factor 1.5) | `duration_ms`, `chorus_hit`, `sections`, `tempo`, `loudness` | Limit the effect of outliers |
| `MinMaxScaler` | `loudness`, `tempo`, `duration_ms`, `key`, `time_signature`, `chorus_hit`, `sections` | Bring features to the 0–1 range |

The outlier bounds (`saved_outlier_bounds`) and the fitted `scaler` are reused for new inputs in the web app, so the same transformations apply at prediction time.

### 3. Train / test split
90% training (5,758 songs) and 10% testing (640 songs), `random_state=42`.

### 4. Models

| Model | Main settings |
|---|---|
| Decision Tree | `max_depth=4` |
| Random Forest | `n_estimators=350` |
| SVM | `kernel='rbf'`, `C=3.5` |
| XGBoost | `n_estimators=350` |
| Bagging | Decision Tree base, `n_estimators=350`, `max_samples=0.6` |

---

## 🏆 Results

Evaluated on the 640-song test set:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **86.1%** | 83.0% | 90.1% | 86.4% |
| Bagging | 85.0% | 82.6% | 87.9% | 85.2% |
| XGBoost | 84.4% | 81.1% | 88.9% | 84.8% |
| SVM | 83.8% | 77.9% | 93.3% | 84.9% |
| Decision Tree | 81.4% | 77.6% | 87.3% | 82.2% |

Random Forest has the best accuracy and F1 score, so it is the model used in the web app. The notebook also plots an accuracy comparison bar chart and a confusion matrix for each model.

---

## 🌐 Web App

The Gradio interface takes the 15 audio features through sliders, a radio button (`mode`) and number fields (`duration_ms`, `chorus_hit`).

How it works:
1. `preprocess_input` applies log transform, outlier capping and scaling to the new song, using the same columns order as training.
2. `predict_hit` sends the processed row to the Random Forest model.
3. The app shows **HIT** or **NOT A HIT** with a confidence percentage.

`demo.launch(share=True)` creates a temporary public link.

---

## 🚀 How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Upload `spotify_hits_10s.csv` to the same folder as the notebook.
3. Install the requirements:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost gradio
   ```
4. Run all cells from top to bottom. The last cell launches the Gradio app and prints the link.

---

## 📁 Project Structure

```
├── spotify.ipynb            # Full notebook: EDA, preprocessing, models, web app
├── spotify_hits_10s.csv     # Dataset
└── README.md
```

---

## ⚠️ Known Limitations and Future Work

- The `MinMaxScaler` is fitted on the whole dataset before the train/test split, which causes slight data leakage. Fitting it on the training set only would be more correct.
- `chorus_hit` is in **seconds** in the dataset, but the web form's default value is `30000`, which looks like milliseconds. It should be about `30`.
- The `time_signature` slider starts at 1, while the dataset also contains 0.
- The model uses audio features only, with no artist popularity or release information.
- Possible improvements: hyperparameter tuning with cross-validation, feature importance analysis, and adding more recent data.

---

## 👥 Team

| Member | Responsibility |
|---|---|
| Person 1 | Data exploration and web interface |
| Person 2 | Preprocessing, Decision Tree and Random Forest, `preprocess_input` |
| Person 3 | SVM, XGBoost and Bagging, evaluation, `predict_hit` |
