# Multi-Modal Used Car Price Prediction & Explainability Framework

An end-to-end machine learning pipeline that combines traditional numerical tabular data with deep semantic text representations extracted via Natural Language Processing (NLP) models to predict used car valuations. 

The framework achieves an R2 score of 0.9306 by fusing structured automobile metrics with high-dimensional embeddings from custom convolutional models and pre-trained deep text extractors.

---

## 🏗️ Project Architecture & Pipeline Flow

The repository is modularized into dedicated execution scripts, running sequentially from raw data ingestion to game-theoretic visual explainability:

* 1_data_preprocessing.py: Ingests raw data, handles cutoffs, extracts mechanical features, and normalizes tabular metrics via RobustScaler.
* 2_extract_deberta.py: Tokenizes and processes string fields to extract global bidirectional contextual text tokens via DistilRoBERTa.
* 3_extract_textcnn.py: Trains a localized 1D Convolutional Neural Network from scratch on automobile phrases, yielding a tight 64-dimensional mechanical descriptor bottleneck.
* 4_extract_llm.py: Taps into Meta LLaMA 1B hidden attention layers to extract parametric market world-knowledge and brand prestige weights.
* 5_model_stacking.py: Fits and serializes your true Level-0 experts alongside your ultimate Level-1 5-estimator Stacking Regressor onto disk as binary weights (.joblib).
* 6_comprehensive_matrix.py: Evaluates and builds the full performance validation results grid testing 5 distinct data scenarios across multiple data-splitting cross-passes.
* 7_explainability_suite.py: Computes macro SHAP feature importance contributions and localized LIME individual instance validation graphs.

---

## ⚙️ Step-by-Step Script Functionality

### 1_data_preprocessing.py
* Tabular Cleanup: Converts alphanumeric currency and mileage entries into numeric values.
* Outlier Truncation: Drops extreme price values outside the 5th and 95th percentiles to preserve model scaling properties.
* Feature Extraction: Extracts engine displacement sizes from string patterns and calculates car_age relative to 2026.
* Text Normalization: Merges sparse string markers into a synthesized textual description string called car_text_profile.
* Robust Scaling: Scales mileage, age, and displacement metrics using a RobustScaler to protect the model boundaries from remaining skewed variances.

### 2_extract_deberta.py & 4_extract_llm.py
* Leverages advanced deep language models to encode global semantic relationships and implicit manufacturer asset hierarchies.
* To avoid the Curse of Dimensionality, high-dimensional outputs (768 for DistilRoBERTa, 2048 for LLaMA 1B) are compressed down to the top 20 Principal Components (PCA), capturing maximum variance without noise.

### 3_extract_textcnn.py
* Trains a 1D Convolutional Neural Network from scratch on local listing text token indexes.
* Utilizes sliding windows of size 2, 3, and 4 tokens to lock onto specialized regional n-gram phrases (Twin-Turbo, AWD Automatic, etc.) mapped directly to target price variables. Exports a tight 64-dimensional latent bottleneck feature matrix.

### 5_model_stacking.py & 6_comprehensive_matrix.py
* Maps out 5 distinct input scenarios across variable test-set partition combinations.
* Fits 5 independent baseline algorithms: Linear Regression, Support Vector Regression (SVR), Random Forest, XGBoost, and LightGBM.
* Fits a Level-1 Stacking Regressor using a 5-Fold out-of-fold cross-validation scheme over a Ridge Regression meta-estimator to create the ultimate champion network.

---

## 📈 Empirical Evaluation Results Matrix

The performance metrics clearly demonstrate how adding NLP text signals radically enhances prediction boundaries over traditional numerical spreadsheets alone:

* Tabular Standard Only (30% Split, LightGBM) -> MAE: 6,347.65 | RMSE: 9,896.07 | R2: 0.8018
* Tabular Standard Only (30% Split, Hybrid Stack) -> MAE: 6,313.18 | RMSE: 9,964.12 | R2: 0.7991
* Isolate: TextCNN Only (30% Split, Hybrid Stack) -> MAE: 4,397.91 | RMSE: 6,377.28 | R2: 0.9177
* Combined Master Megafeature (30% Split, LightGBM) -> MAE: 3,988.73 | RMSE: 6,021.19 | R2: 0.9266
* Combined Master Megafeature (30% Split, Hybrid Stack) -> MAE: 3,786.95 | RMSE: 5,855.11 | R2: 0.9306

### Key Findings
1. Multimodal Synergy: Transitioning from numbers-only metrics to a multimodal feature space decreases the Mean Absolute Error (MAE) by $2,526.23 per vehicle, while elevating variance explanation up to 93.06%.
2. The TextCNN Phenomenon: In pure isolation, running models completely blind to odometer or age features using only the custom TextCNN bottleneck layer yields an R2 of 0.9177. This confirms that localized, custom-trained mechanical descriptors carry immense financial signal.

---

## 🔬 Model Explainability & Visual Diagnostics (7_explainability_suite.py)

The pipeline explicitly opens up its black-box mechanics by outputting 5 publication-ready diagnostic charts into figures/:

1. fig1_prediction_fit.png (Prediction Scatter): Visual regression tracking predicted vs true valuations along a tight 45-degree absolute fit trajectory.
2. fig2_residual_distribution.png (Error Histogram): Statistical proof showing that framework prediction deviations conform to a healthy, unbiased Gaussian bell curve centered right at zero.
3. fig3_shap_importance.png (Global SHAP Breakdown): Game-theoretic feature weights ranking feature categories globally. It proves custom text-mined features yield higher predictive signal than physical automobile attributes.
4. fig4_lime_local_explanation.png (LIME Instance Check): Dissects a localized snapshot explanation for an individual vehicle, highlighting exactly which threshold conditions drove its precise market valuation without text overlap errors.
5. fig5_heteroscedasticity_age.png (Error Variance Check): Validates error variance equality (homoscedasticity) across the timeline of scaled vehicle lifespans.

---

## 🚀 Execution Guide

1. Clone the repository and establish your python virtual environment layout:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Place your raw data file inside data/raw/used_cars.csv.

3. Sequentially process the architecture files to train feature stores, export binaries, and map explainability assets:

   python3 1_data_preprocessing.py

   python3 2_extract_deberta.py

   python3 3_extract_textcnn.py

   python3 4_extract_llm.py

   python3 5_model_stacking.py

   python3 6_comprehensive_matrix.py
   
   python3 7_explainability_suite.py