"""
Model 1 (FIXED): Cutoff Prediction Model
Predicts 2025 closing ranks WITHOUT data leakage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


class CutoffPredictionModelFixed:
    def __init__(self, data_path='processed_data/josaa_ml_ready.csv'):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.best_model = None
        self.feature_columns = None
        self.output_dir = Path('models')
        self.output_dir.mkdir(exist_ok=True)
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        """Load preprocessed data"""
        print("📂 Loading preprocessed data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} records")
        return self.df
    
    def create_lag_features(self):
        """
        Create LAG features (previous year's data) - CRITICAL FIX!
        This prevents data leakage by using ONLY past information
        """
        print("\n🔧 Creating proper lag features (no data leakage)...")
        
        # Sort by group and year
        self.df = self.df.sort_values(['Institute', 'Academic Program Name', 'Seat Type', 'Gender', 'Round', 'Year'])
        
        group_cols = ['Institute', 'Academic Program Name', 'Seat Type', 'Gender', 'Round']
        
        # Previous year's closing rank (LAG 1)
        self.df['prev_year_closing_rank'] = self.df.groupby(group_cols)['Closing Rank'].shift(1)
        
        # Previous year's opening rank
        self.df['prev_year_opening_rank'] = self.df.groupby(group_cols)['Opening Rank'].shift(1)
        
        # 2 years ago closing rank (LAG 2)
        self.df['prev_2year_closing_rank'] = self.df.groupby(group_cols)['Closing Rank'].shift(2)
        
        # Average of last 2 years (using ONLY past data)
        self.df['avg_past_2yr'] = (self.df['prev_year_closing_rank'] + self.df['prev_2year_closing_rank']) / 2
        
        # Year-over-year change from previous data
        self.df['prev_yoy_change'] = self.df['prev_year_closing_rank'] - self.df['prev_2year_closing_rank']
        
        # Drop rows where we don't have lag features
        before = len(self.df)
        self.df = self.df.dropna(subset=['prev_year_closing_rank'])
        print(f"✓ Removed {before - len(self.df)} rows without lag features")
        print(f"✓ Remaining: {len(self.df)} records")
        
        # Fill missing 2-year lag with 1-year lag
        self.df['prev_2year_closing_rank'] = self.df['prev_2year_closing_rank'].fillna(self.df['prev_year_closing_rank'])
        self.df['avg_past_2yr'] = self.df['avg_past_2yr'].fillna(self.df['prev_year_closing_rank'])
        self.df['prev_yoy_change'] = self.df['prev_yoy_change'].fillna(0)
        
        return self.df
    
    def prepare_features(self):
        """Prepare features WITHOUT current year's closing rank"""
        print("\n🔧 Preparing features (no leakage)...")
        
        # CRITICAL: Use only PAST data and contextual features
        self.feature_columns = [
            'Year',
            'Round',
            'Institute_Code',
            'Institute_Type_Code',
            'Academic Program Name_Code',
            'Branch_Category_Code',
            'Quota_Code',
            'Seat Type_Code',
            'Gender_Code',
            'NIRF_Rank',
            'Years_Since_2020',
            'Is_Covid_Year',
            'Round_Progression',
            # LAG features (previous years only)
            'prev_year_closing_rank',
            'prev_2year_closing_rank',
            'prev_year_opening_rank',
            'avg_past_2yr',
            'prev_yoy_change',
        ]
        
        # Keep only columns that exist
        self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
        
        print(f"✓ Features ({len(self.feature_columns)}): {self.feature_columns}")
        print(f"✓ Target: Closing Rank")
        
        return self.feature_columns
    
    def split_data(self, test_year=2024):
        """Time-based split"""
        print(f"\n📊 Splitting data (test year: {test_year})...")
        
        train_df = self.df[self.df['Year'] < test_year].copy()
        test_df = self.df[self.df['Year'] == test_year].copy()
        
        X_train = train_df[self.feature_columns]
        y_train = train_df['Closing Rank']
        X_test = test_df[self.feature_columns]
        y_test = test_df['Closing Rank']
        
        print(f"✓ Training set: {len(X_train)} records (years {train_df['Year'].min()}-{train_df['Year'].max()})")
        print(f"✓ Test set: {len(X_test)} records (year {test_year})")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        print("\n🤖 Training multiple models...")
        
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1),
            'LightGBM': LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        }
        
        for name, model in models_to_train.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("\n📏 Evaluating models...")
        print("="*80)
        print(f"{'Model':<25} {'MAE':<12} {'RMSE':<12} {'R²':<10} {'MAPE':<10}")
        print("="*80)
        
        results = {}
        best_mae = float('inf')
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            # Ensure non-negative predictions
            y_pred = np.maximum(y_pred, 1)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape}
            
            print(f"{name:<25} {mae:<12.2f} {rmse:<12.2f} {r2:<10.4f} {mape:<10.2f}%")
            
            if mae < best_mae:
                best_mae = mae
                self.best_model_name = name
                self.best_model = model
        
        print("="*80)
        print(f"\n🏆 Best Model: {self.best_model_name} (MAE: {best_mae:.2f})")
        
        results_df = pd.DataFrame(results).T
        results_df.to_csv(self.results_dir / 'model_comparison.csv')
        
        return results
    
    def feature_importance(self):
        """Analyze feature importance"""
        print("\n🎯 Analyzing feature importance...")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Features:")
            print(feature_df.head(10).to_string(index=False))
            
            feature_df.to_csv(self.results_dir / 'feature_importance.csv', index=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_df.head(10), x='Importance', y='Feature')
            plt.title(f'Top 10 Feature Importance - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'feature_importance.png', dpi=100)
            plt.close()
            print(f"✓ Plot saved")
            
            return feature_df
        return None
    
    def plot_predictions(self, X_test, y_test):
        """Diagnostic plots"""
        print("\n📈 Creating plots...")
        
        y_pred = self.best_model.predict(X_test)
        y_pred = np.maximum(y_pred, 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Closing Rank')
        axes[0].set_ylabel('Predicted Closing Rank')
        axes[0].set_title(f'Actual vs Predicted - {self.best_model_name}')
        axes[0].grid(True, alpha=0.3)
        
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'prediction_diagnostics.png', dpi=100)
        plt.close()
        print(f"✓ Plots saved")
    
    def predict_2025(self):
        """Predict 2025 cutoffs using 2024 as the lag"""
        print("\n🔮 Predicting 2025 cutoffs...")
        
        # Get 2024 data (this becomes 'prev_year' for 2025)
        df_2024 = self.df[self.df['Year'] == 2024].copy().reset_index(drop=True)
        df_2023 = self.df[self.df['Year'] == 2023].copy()
        
        if len(df_2024) == 0:
            print("⚠️ No 2024 data available for prediction")
            return None
        
        # Create 2025 prediction template from 2024
        df_2025 = df_2024.copy()
        df_2025['Year'] = 2025
        df_2025['Years_Since_2020'] = 5
        df_2025['Is_Covid_Year'] = 0
        
        # Update lag features for 2025 prediction
        df_2025['prev_year_closing_rank'] = df_2024['Closing Rank'].values
        df_2025['prev_year_opening_rank'] = df_2024['Opening Rank'].values
        
        # 2-year lag = 2023 data (need to merge)
        merge_cols = ['Institute', 'Academic Program Name', 'Seat Type', 'Gender', 'Round']
        df_2023_subset = df_2023[merge_cols + ['Closing Rank']].rename(columns={'Closing Rank': 'cr_2023'})
        
        # Drop duplicates in df_2023_subset to avoid row multiplication during merge
        df_2023_subset = df_2023_subset.drop_duplicates(subset=merge_cols, keep='first')
        
        # Use left merge - only keep rows from df_2025
        df_2025 = df_2025.merge(df_2023_subset, on=merge_cols, how='left')
        df_2025['prev_2year_closing_rank'] = df_2025['cr_2023'].fillna(df_2025['prev_year_closing_rank'])
        
        df_2025['avg_past_2yr'] = (df_2025['prev_year_closing_rank'] + df_2025['prev_2year_closing_rank']) / 2
        df_2025['prev_yoy_change'] = df_2025['prev_year_closing_rank'] - df_2025['prev_2year_closing_rank']
        
        # Predict
        X_2025 = df_2025[self.feature_columns]
        predictions_2025 = self.best_model.predict(X_2025)
        predictions_2025 = np.maximum(predictions_2025, 1)
        
        # Create predictions dataframe - use df_2025's data (which is properly aligned)
        predictions_df = df_2025[['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender', 'Round']].copy()
        predictions_df['Closing_Rank_2024'] = df_2025['prev_year_closing_rank'].values  # This is 2024's closing rank
        predictions_df['Predicted_Closing_Rank_2025'] = predictions_2025.astype(int)
        predictions_df['Change'] = (predictions_df['Predicted_Closing_Rank_2025'] - predictions_df['Closing_Rank_2024']).astype(int)
        predictions_df['Change_Percent'] = ((predictions_df['Change'] / predictions_df['Closing_Rank_2024']) * 100).round(2)
        
        predictions_df.to_csv(self.results_dir / 'predictions_2025.csv', index=False)
        print(f"✓ Saved {len(predictions_df)} 2025 predictions to {self.results_dir / 'predictions_2025.csv'}")
        
        # Show top IIT predictions
        iit_preds = predictions_df[predictions_df['Institute'].str.contains('Indian Institute of Technology', case=False, na=False)]
        if len(iit_preds) > 0:
            print("\n📋 Sample 2025 Predictions (IITs):")
            print(iit_preds.head(10).to_string(index=False))
        else:
            print("\n📋 Sample 2025 Predictions:")
            print(predictions_df.head(10).to_string(index=False))
        
        return predictions_df
    
    def save_model(self):
        """Save best model"""
        model_file = self.output_dir / 'cutoff_prediction_model.pkl'
        joblib.dump({
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_columns': self.feature_columns
        }, model_file)
        print(f"\n💾 Model saved to {model_file}")
    
    def run_pipeline(self):
        """Execute full pipeline"""
        print("="*60)
        print("🤖 CUTOFF PREDICTION MODEL (FIXED - NO LEAKAGE)")
        print("="*60)
        
        self.load_data()
        self.create_lag_features()
        self.prepare_features()
        X_train, X_test, y_train, y_test = self.split_data(test_year=2024)
        self.train_models(X_train, y_train)
        results = self.evaluate_models(X_test, y_test)
        self.feature_importance()
        self.plot_predictions(X_test, y_test)
        self.predict_2025()
        self.save_model()
        
        print("\n" + "="*60)
        print("✅ MODEL 1 COMPLETE!")
        print("="*60)
        
        return results


if __name__ == "__main__":
    model = CutoffPredictionModelFixed()
    model.run_pipeline()
