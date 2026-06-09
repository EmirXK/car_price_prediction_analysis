import pandas as pd
import numpy as np
import os
import joblib  # Standard tool for serializing massive ML pipelines
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

def main():
    # Ensure serialization folder exists
    os.makedirs('models', exist_ok=True)

    # 1. Load your feature stores
    print("Loading all tabular and cached model data features...")
    df_tab = pd.read_parquet('data/processed/base_tabular.parquet')
    deberta_feats = np.load('data/processed/deberta_embeddings.npy')
    textcnn_feats = np.load('data/processed/textcnn_features.npy')
    llm_feats = np.load('data/processed/llm_features.npy')
    
    # 2. Dimensionality Reduction (PCA)
    print("Compressing high-dimensional NLP spaces via PCA components...")
    pca_deberta = PCA(n_components=20, random_state=42)
    deberta_reduced = pca_deberta.fit_transform(deberta_feats)
    
    pca_llm = PCA(n_components=20, random_state=42)
    llm_reduced = pca_llm.fit_transform(llm_feats)
    
    # Save the fitted PCA transformers so you can transform future evaluation data
    joblib.dump(pca_deberta, 'models/pca_deberta.joblib')
    joblib.dump(pca_llm, 'models/pca_llm.joblib')
    
    # 3. Create DataFrame Wrappers
    df_deberta = pd.DataFrame(deberta_reduced, columns=[f'deberta_pc_{i}' for i in range(20)])
    df_llm = pd.DataFrame(llm_reduced, columns=[f'llama_pc_{i}' for i in range(20)])
    df_textcnn = pd.DataFrame(textcnn_feats, columns=[f'textcnn_dim_{i}' for i in range(64)])
    
    # 4. Synthesize Master Matrix
    X = pd.concat([
        df_tab.drop(columns=['price', 'car_id']), 
        df_deberta, 
        df_textcnn,
        df_llm
    ], axis=1)
    y = df_tab['price']
    X.fillna(0, inplace=True)

    # 5. Partition Data (Matching your champion 30% validation hold)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Master Matrix Synced! Total feature variables evaluated: {X_train.shape[1]}")

    # 6. Define and Train Individual Algorithms
    models = {
        "linear_model": LinearRegression(),
        "svm_model": SVR(kernel='rbf', C=25000, epsilon=0.1),
        "random_forest_model": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "xgboost_model": XGBRegressor(n_estimators=100, learning_rate=0.08, random_state=42),
        "lightgbm_model": LGBMRegressor(n_estimators=120, learning_rate=0.08, random_state=42, verbose=-1)
    }

    print(f"\n{'Model Architecture':<25} | {'MAE':<12} | {'RMSE':<12} | {'R2 Score':<10}")
    print("-" * 72)
    
    for filename, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"{filename:<25} | ${mean_absolute_error(y_test, preds):<11,.2f} | ${root_mean_squared_error(y_test, preds):<11,.2f} | {r2_score(y_test, preds):<10.4f}")
        
        # Serialize model to disk immediately after evaluation
        joblib.dump(model, f'models/{filename}.joblib')
        print(f"   [Saved to models/{filename}.joblib]")

    # 7. Expanded Level-1 Hybrid Ensemble Stacking Super-Learner (5 Experts)
    print("\n" + "="*50)
    print("Fitting Expanded Level-1 Stacking Hybrid Regressor...")
    print("="*50)
    
    estimators = [
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.08, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=120, learning_rate=0.08, random_state=42, verbose=-1)),
        ('svr', SVR(kernel='rbf', C=25000, epsilon=0.1))
    ]
    
    stacked_ensemble = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=5.0),
        cv=5,
        n_jobs=-1
    )
    
    stacked_ensemble.fit(X_train, y_train)
    ens_preds = stacked_ensemble.predict(X_test)
    
    print(f"\n{'FINAL HYBRID STACK':<25} | ${mean_absolute_error(y_test, ens_preds):<11,.2f} | ${root_mean_squared_error(y_test, ens_preds):<11,.2f} | {r2_score(y_test, ens_preds):<10.4f}")
    
    # Save your champion hybrid ensemble configuration
    joblib.dump(stacked_ensemble, 'models/final_stacked_ensemble.joblib')
    print("   [Saved to models/final_stacked_ensemble.joblib]")

if __name__ == "__main__":
    main()