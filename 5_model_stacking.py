import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_validate
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

    # 6. Define Individual Algorithms
    models = {
        "linear_model": LinearRegression(),
        "svm_model": SVR(kernel='rbf', C=25000, epsilon=0.1),
        "random_forest_model": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "xgboost_model": XGBRegressor(n_estimators=100, learning_rate=0.08, random_state=42),
        "lightgbm_model": LGBMRegressor(n_estimators=120, learning_rate=0.08, random_state=42, verbose=-1)
    }

    print("\n" + "="*85)
    print("Evaluating Individual Base Models using 10-Fold Cross-Validation...")
    print("="*85)
    print(f"{'Model Architecture':<23} | {'10-Fold CV R²':<13} | {'Test MAE':<12} | {'Test RMSE':<12} | {'Test R²':<10}")
    print("-" * 85)
    
    for filename, model in models.items():
        # Execute 10-Fold Cross-Validation on the training split to get robust validation metrics
        cv_results = cross_validate(
            model, X_train, y_train, 
            cv=10, 
            scoring='r2', 
            n_jobs=-1
        )
        mean_cv_r2 = np.mean(cv_results['test_score'])
        
        # Fit on full training data partition to evaluate on holdout test set
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        test_mae = mean_absolute_error(y_test, preds)
        test_rmse = root_mean_squared_error(y_test, preds)
        test_r2 = r2_score(y_test, preds)
        
        print(f"{filename:<23} | {mean_cv_r2:<13.4f} | ${test_mae:<11,.2f} | ${test_rmse:<11,.2f} | {test_r2:<10.4f}")
        
        # Serialize model to disk immediately after evaluation
        joblib.dump(model, f'models/{filename}.joblib')

    # 7. Expanded Level-1 Hybrid Ensemble Stacking Super-Learner (5 Experts)
    print("\n" + "="*85)
    print("Fitting Expanded Level-1 Stacking Hybrid Regressor (Using 10-Fold CV)...")
    print("="*85)
    
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
        cv=10,  # Switched from 5-fold to 10-fold cross-validation layer
        n_jobs=-1
    )
    
    # Train the ensemble and use 10-fold internal CV splits for the meta-learner feature synthesis
    stacked_ensemble.fit(X_train, y_train)
    ens_preds = stacked_ensemble.predict(X_test)
    
    ens_mae = mean_absolute_error(y_test, ens_preds)
    ens_rmse = root_mean_squared_error(y_test, ens_preds)
    ens_r2 = r2_score(y_test, ens_preds)
    
    print(f"\n{'FINAL HYBRID STACK':<23} | {'[Meta Trained]':<13} | ${ens_mae:<11,.2f} | ${ens_rmse:<11,.2f} | {ens_r2:<10.4f}")
    
    # Save your champion hybrid ensemble configuration
    joblib.dump(stacked_ensemble, 'models/final_stacked_ensemble.joblib')
    print("   [Saved to models/final_stacked_ensemble.joblib]")

if __name__ == "__main__":
    main()