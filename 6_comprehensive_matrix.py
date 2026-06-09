import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge # Added Linear Regression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor # Added Random Forest
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tqdm import tqdm

def main():
    # Ensure directory layout
    os.makedirs('results', exist_ok=True)
    
    print("Loading distributed modal storage blocks...")
    df_tab = pd.read_parquet('data/processed/base_tabular.parquet')
    deberta_feats = np.load('data/processed/deberta_embeddings.npy')
    textcnn_feats = np.load('data/processed/textcnn_features.npy')
    llm_feats = np.load('data/processed/llm_features.npy')
    
    # Pre-compress heavy representations for processing sanity
    print("Executing foundational PCA reductions...")
    pca_deb = PCA(n_components=20, random_state=42).fit_transform(deberta_feats)
    pca_llm = PCA(n_components=20, random_state=42).fit_transform(llm_feats)
    
    # 1. Map Out Data Feature Subsets
    X_tabular = df_tab.drop(columns=['price', 'car_id']).fillna(0)
    y = df_tab['price']
    
    data_scenarios = {
        "Tabular Standard Only": X_tabular,
        "Isolate: DistilRoBERTa Only": pd.DataFrame(pca_deb, index=df_tab.index),
        "Isolate: TextCNN Only": pd.DataFrame(textcnn_feats, index=df_tab.index),
        "Isolate: LLaMA 1B Only": pd.DataFrame(pca_llm, index=df_tab.index),
        "Combined Master Megafeature": pd.concat([
            X_tabular, 
            pd.DataFrame(pca_deb, columns=[f'deb_{i}' for i in range(20)], index=df_tab.index),
            pd.DataFrame(textcnn_feats, columns=[f'cnn_{i}' for i in range(64)], index=df_tab.index),
            pd.DataFrame(pca_llm, columns=[f'llm_{i}' for i in range(20)], index=df_tab.index)
        ], axis=1)
    }
    
    # 2. Setup Evaluation Splitting Variations
    test_splits = {
        "20% Test Split": 0.2,
        "30% Test Split": 0.3
    }
    
    # Container for all rows
    matrix_results = []
    
    # Total combinations tracking loop: Now 6 models evaluated per pass (5 base + 1 stack)
    total_runs = len(data_scenarios) * len(test_splits) * 6 
    progress_bar = tqdm(total=total_runs, desc="Empirical Grid Matrix Evaluation")
    
    for scenario_name, X_data in data_scenarios.items():
        # Clean structural alignment copy
        X_curr = X_data.fillna(0).copy()
        
        for split_label, split_val in test_splits.items():
            # Executing Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_curr, y, test_size=split_val, random_state=42
            )
            
            # Dynamic Estimator Instantiations (Expanded Architecture List)
            models = {
                "Baseline: Linear Regression": LinearRegression(),
                "Algorithm-1: SVR (SVM)": SVR(kernel='rbf', C=25000, epsilon=0.1),
                "Ensemble: Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "Algorithm-5: XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.08, random_state=42),
                "Algorithm-5: LightGBM": LGBMRegressor(n_estimators=120, learning_rate=0.08, random_state=42, verbose=-1)
            }
            
            # Evaluate Base Level-0 Architectures
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                matrix_results.append({
                    "Feature Scenario": scenario_name,
                    "Data Split Pattern": split_label,
                    "Model Target": model_name,
                    "MAE ($)": round(mean_absolute_error(y_test, preds), 2),
                    "RMSE ($)": round(root_mean_squared_error(y_test, preds), 2),
                    "R2 Score": round(r2_score(y_test, preds), 4)
                })
                progress_bar.update(1)
                
            # Evaluate Level-1 Ensemble Stacking System (Expanded to 5 Base Estimators)
            estimators = [
                ('lr', LinearRegression()),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.08, random_state=42)),
                ('lgbm', LGBMRegressor(n_estimators=120, learning_rate=0.08, random_state=42, verbose=-1)),
                ('svr', SVR(kernel='rbf', C=25000, epsilon=0.1))
            ]
            
            stack = StackingRegressor(
                estimators=estimators, 
                final_estimator=Ridge(alpha=5.0), 
                cv=5, 
                n_jobs=-1
            )
            stack.fit(X_train, y_train)
            stack_preds = stack.predict(X_test)
            
            matrix_results.append({
                "Feature Scenario": scenario_name,
                "Data Split Pattern": split_label,
                "Model Target": "FINAL HYBRID ENSEMBLE STACK",
                "MAE ($)": round(mean_absolute_error(y_test, stack_preds), 2),
                "RMSE ($)": round(root_mean_squared_error(y_test, stack_preds), 2),
                "R2 Score": round(r2_score(y_test, stack_preds), 4)
            })
            progress_bar.update(1)

    progress_bar.close()
    
    # 3. Transpile list structures to formal DataFrames and save
    df_results = pd.DataFrame(matrix_results)
    df_results.to_csv('results/structural_empirical_results.csv', index=False)
    
    print("\n" + "="*60)
    print("SUCCESS: Expanded Statistical Performance Grid Written to Disk!")
    print("Path: results/structural_empirical_results.csv")
    print("="*60)
    
    # Print out a quick snapshot preview of the top performers
    print("\nPreview of Top Performing Scenarios (Sorted by Highest R2):")
    print(df_results.sort_values(by="R2 Score", ascending=False).head(12).to_string(index=False))

if __name__ == "__main__":
    main()