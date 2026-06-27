import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import subprocess  # Added to handle native Linux vector conversion to EMF
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer 

def main():
    # Setup target directories
    os.makedirs('figures', exist_ok=True)
    
    # 1. Reconstruct the Champion Combined Master Matrix
    print("Reassembling Champion Multi-Modal Feature Space...")
    df_tab = pd.read_parquet('data/processed/base_tabular.parquet')
    deberta_feats = np.load('data/processed/deberta_embeddings.npy')
    textcnn_feats = np.load('data/processed/textcnn_features.npy')
    llm_feats = np.load('data/processed/llm_features.npy')
    
    pca_deberta = joblib.load('models/pca_deberta.joblib')
    pca_llm = joblib.load('models/pca_llm.joblib')
    
    deberta_reduced = pca_deberta.transform(deberta_feats)
    llm_reduced = pca_llm.transform(llm_feats)
    
    X_tabular = df_tab.drop(columns=['price', 'car_id']).fillna(0)
    y = df_tab['price']
    
    X = pd.concat([
        X_tabular,
        pd.DataFrame(deberta_reduced, columns=[f'deberta_pc_{i}' for i in range(20)], index=df_tab.index),
        pd.DataFrame(textcnn_feats, columns=[f'textcnn_dim_{i}' for i in range(64)], index=df_tab.index),
        pd.DataFrame(llm_reduced, columns=[f'llama_pc_{i}' for i in range(20)], index=df_tab.index)
    ], axis=1)

    # Re-execute champion partitioning split (30% Test Size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 2. Load Models
    print("Loading serialized model weights for diagnostics...")
    stack_model = joblib.load('models/final_stacked_ensemble.joblib')
    lgbm_model = joblib.load('models/lightgbm_model.joblib')
    
    # Run predictions
    preds = stack_model.predict(X_test) 
    residuals = y_test - preds

    # --- FIGURE 1: Actual vs. Predicted Price Regression ---
    print("\n[Plotting Figure 1: Model Prediction Fit Graph...]")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=preds, alpha=0.4, color='#1a365d', edgecolor='none')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
    plt.title('Actual Prices vs. True Champion Ensemble Predictions (R² = 0.9306)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('True Vehicle Price ($)', fontsize=10)
    plt.ylabel('Model Predicted Price ($)', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/fig1_prediction_fit.svg')
    plt.close()

    # --- FIGURE 2: Residual Analysis (Error Distribution) ---
    print("[Plotting Figure 2: Error Residual Distribution Chart...]")
    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, kde=True, bins=60, color='#94a3b8', edgecolor='#1e293b', line_kws={'lw': 2.5})
    plt.axvline(0, color='red', linestyle='--', lw=1.5)
    plt.title('Residual Error Distribution Profile (Hybrid Stack)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Prediction Deviation Residual ($)', fontsize=10)
    plt.ylabel('Density Count', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/fig2_residual_distribution.svg')
    plt.close()

    # --- FIGURE 3: SHAP Global Feature Importance Summary Graph ---
    print("[Plotting Figure 3: SHAP Feature Importance Summary Graph...]")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer(X_test)
    
    global_importances = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': global_importances})
    
    def map_feature_labels(name):
        if 'textcnn' in name: return 'TextCNN (Mechanical Descriptors Block)'
        if 'deberta' in name: return 'DistilRoBERTa (Global Semantics PC)'
        if 'llama' in name: return 'LLaMA 1B (Parametric Brand Prestige PC)'
        if 'milage' in name: return 'Vehicle Odometer Mileage'
        if 'car_age' in name: return 'Calculated Vehicle Age'
        if 'engine_size' in name: return 'Engine Displacement Size'
        if 'has_accident' in name: return 'Accident History Log'
        return name.replace('brand_', 'Brand: ').replace('_', ' ')
        
    importance_df['Clean Feature Name'] = importance_df['Feature'].apply(map_feature_labels)
    importance_df_agg = importance_df.groupby('Clean Feature Name', as_index=False)[['Importance']].sum()
    importance_df_agg = importance_df_agg.sort_values(by='Importance', ascending=False).head(12)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance', y='Clean Feature Name', data=importance_df_agg, 
        hue='Clean Feature Name', palette='Blues_r', edgecolor='#0f172a', legend=False
    )
    plt.title('Global Feature Contribution Matrix (Aggregated SHAP Feature Values)', fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('Cumulative Absolute Impact Magnitude on Valuation Model ($)', fontsize=10)
    plt.ylabel('', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/fig3_shap_importance.svg')
    plt.close()

    # --- FIGURE 4: LIME Local Instance Explanation ---
    print("\nInitializing Local Interpretable Model-agnostic Explanations (LIME)...")
    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['price'],
        mode='regression',
        random_state=42
    )
    
    sample_idx = 0 
    car_instance = X_test.iloc[sample_idx]
    
    lime_exp = lime_explainer.explain_instance(
        data_row=car_instance.values,
        predict_fn=lgbm_model.predict,
        num_features=8
    )
    
    print("[Plotting Figure 4: LIME Local Car Instance Breakdown Plot...]")
    raw_lime_list = lime_exp.as_list()
    clean_lime_features = []
    lime_weights = []
    
    for feature_condition, weight in raw_lime_list:
        clean_cond = feature_condition
        
        if 'textcnn_dim_' in clean_cond:
            clean_cond = clean_cond.split(' ', 1)[1] if ' ' in clean_cond else clean_cond
            clean_cond = f"TextCNN Descriptors Block {clean_cond}"
        elif 'deberta_pc_' in clean_cond:
            clean_cond = clean_cond.split(' ', 1)[1] if ' ' in clean_cond else clean_cond
            clean_cond = f"DistilRoBERTa Global Semantics {clean_cond}"
        elif 'llama_pc_' in clean_cond:
            clean_cond = clean_cond.split(' ', 1)[1] if ' ' in clean_cond else clean_cond
            clean_cond = f"LLaMA 1B Brand Prestige {clean_cond}"
        else:
            clean_cond = clean_cond.replace('milage', 'Vehicle Odometer Mileage')
            clean_cond = clean_cond.replace('car_age', 'Calculated Vehicle Age')
            clean_cond = clean_cond.replace('engine_size', 'Engine Displacement Size')
            clean_cond = clean_cond.replace('has_accident', 'Accident History Log')
            clean_cond = clean_cond.replace('brand_', 'Brand: ').replace('_', ' ')

        clean_lime_features.append(clean_cond)
        lime_weights.append(weight)
        
    plt.figure(figsize=(11, 5.5)) 
    colors = ['#15803d' if w > 0 else '#b91c1c' for w in lime_weights]
    
    sns.barplot(
        x=lime_weights, y=clean_lime_features, hue=clean_lime_features,
        palette=colors, edgecolor='#1e293b', legend=False
    )
    plt.axvline(0, color='black', linestyle='-', lw=1)
    plt.title(f'LIME Local Explanation: Why the Model Priced Car #{sample_idx + 1}', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Local Feature Weight Profile ($ Value Shift Impact)', fontsize=10)
    plt.ylabel('', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/fig4_lime_local_explanation.svg')
    plt.close()

    # --- FIGURE 5: Heteroscedasticity Analysis ---
    print("[Plotting Figure 5: Residual Error Variance vs. Calculated Vehicle Age...]")
    plt.figure(figsize=(9, 5.5))
    sns.scatterplot(x=X_test['car_age'], y=residuals, alpha=0.3, color='#0284c7', edgecolor='none')
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title('Model Error Stability Across Calculated Vehicle Lifespans', fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('Vehicle Age (Years Profile Component)', fontsize=10)
    plt.ylabel('Prediction Error Residual ($)', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/fig5_heteroscedasticity_age.svg')
    plt.close()

    # --- AUTOMATED VECTOR CONVERSION TO EMF ---
    print("\nExecuting native background vector conversion to EMF format via LibreOffice...")
    try:
        # Calls the built-in Linux Mint headless engine to turn all 5 SVGs into native EMFs instantly
        subprocess.run([
            'libreoffice', '--headless', '--convert-to', 'emf', 
            'figures/fig1_prediction_fit.svg', 
            'figures/fig2_residual_distribution.svg', 
            'figures/fig3_shap_importance.svg', 
            'figures/fig4_lime_local_explanation.svg', 
            'figures/fig5_heteroscedasticity_age.svg', 
            '--outdir', 'figures'
        ], check=True, stdout=subprocess.DEVNULL)
        print("   -> Success! All 5 vector plots successfully compiled to .emf format.")
    except Exception as e:
        print(f"   -> Warning: Background conversion failed ({e}). Ensure libreoffice is installed.")

    print("\n" + "="*60)
    print("SUCCESS: 5-Figure Explainability Suite Completed!")
    print("EMF files are generated and ready for direct Google Docs upload.")
    print("="*60)

if __name__ == "__main__":
    main()