"""
Model 2 (FIXED): Admission Probability Classifier
Predicts admission probability WITHOUT data leakage
Uses HISTORICAL data to predict future admissions
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class AdmissionProbabilityModelFixed:
    def __init__(self, data_path='processed_data/josaa_ml_ready.csv'):
        self.data_path = data_path
        self.df = None
        self.training_data = None
        self.models = {}
        self.best_model = None
        self.feature_columns = None
        self.output_dir = Path('models')
        self.output_dir.mkdir(exist_ok=True)
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        print("📂 Loading preprocessed data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} records")
        return self.df
    
    def create_realistic_training_data(self):
        """
        Create REALISTIC training data:
        - Use PREVIOUS year's data to predict CURRENT year admission
        - Don't use current year's closing rank as feature (leakage!)
        """
        print("\n🔧 Creating realistic admission dataset...")
        
        # Sort data
        self.df = self.df.sort_values(['Institute', 'Academic Program Name', 'Seat Type', 'Gender', 'Round', 'Year'])
        
        group_cols = ['Institute', 'Academic Program Name', 'Seat Type', 'Gender', 'Round']
        
        # Add previous year's closing rank (for use as feature)
        self.df['prev_year_closing_rank'] = self.df.groupby(group_cols)['Closing Rank'].shift(1)
        self.df['prev_year_opening_rank'] = self.df.groupby(group_cols)['Opening Rank'].shift(1)
        
        # Drop rows without previous year data
        self.df = self.df.dropna(subset=['prev_year_closing_rank'])
        print(f"✓ Records with previous year data: {len(self.df)}")
        
        training_records = []
        
        # For each record, create training scenarios
        for idx, row in self.df.iterrows():
            actual_closing = row['Closing Rank']
            prev_closing = row['prev_year_closing_rank']
            prev_opening = row['prev_year_opening_rank']
            
            if pd.isna(actual_closing) or actual_closing <= 0:
                continue
            
            # Create scenarios: students with various ranks apply
            # Generate ranks around the previous year's closing rank
            scenarios = [
                int(prev_closing * 0.5),   # Much better rank
                int(prev_closing * 0.7),   # Better rank
                int(prev_closing * 0.85),  # Slightly better
                int(prev_closing * 1.0),   # At previous closing
                int(prev_closing * 1.15),  # Slightly worse
                int(prev_closing * 1.3),   # Worse
                int(prev_closing * 1.5),   # Much worse
            ]
            
            for student_rank in scenarios:
                if student_rank <= 0:
                    continue
                
                # Determine if admitted based on ACTUAL closing rank
                admitted = 1 if student_rank <= actual_closing else 0
                
                training_records.append({
                    'student_rank': student_rank,
                    'prev_year_closing_rank': prev_closing,
                    'prev_year_opening_rank': prev_opening,
                    'rank_vs_prev_closing': student_rank - prev_closing,
                    'rank_ratio_prev': student_rank / prev_closing if prev_closing > 0 else 1.0,
                    'prev_closing_opening_ratio': prev_closing / prev_opening if prev_opening > 0 else 1.0,
                    'year': row['Year'],
                    'round': row['Round'],
                    'institute_code': row['Institute_Code'],
                    'institute_type_code': row['Institute_Type_Code'],
                    'branch_code': row['Academic Program Name_Code'],
                    'branch_category_code': row['Branch_Category_Code'],
                    'quota_code': row['Quota_Code'],
                    'seat_type_code': row['Seat Type_Code'],
                    'gender_code': row['Gender_Code'],
                    'nirf_rank': row['NIRF_Rank'],
                    'round_progression': row['Round_Progression'],
                    'admitted': admitted
                })
        
        self.training_data = pd.DataFrame(training_records)
        print(f"✓ Created {len(self.training_data)} training records")
        print(f"  - Admitted: {(self.training_data['admitted'] == 1).sum()}")
        print(f"  - Not Admitted: {(self.training_data['admitted'] == 0).sum()}")
        
        return self.training_data
    
    def prepare_features(self):
        """Define realistic feature columns - NO CURRENT YEAR DATA"""
        self.feature_columns = [
            'student_rank',
            'prev_year_closing_rank',
            'prev_year_opening_rank',
            'rank_vs_prev_closing',
            'rank_ratio_prev',
            'prev_closing_opening_ratio',
            'year',
            'round',
            'institute_code',
            'institute_type_code',
            'branch_code',
            'branch_category_code',
            'quota_code',
            'seat_type_code',
            'gender_code',
            'nirf_rank',
            'round_progression'
        ]
        print(f"\n✓ Features ({len(self.feature_columns)}): {self.feature_columns}")
    
    def split_data(self):
        """Split with stratification"""
        X = self.training_data[self.feature_columns]
        y = self.training_data['admitted']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Split:")
        print(f"  Training: {len(X_train)} records")
        print(f"  Test: {len(X_test)} records")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple classification models"""
        print("\n🤖 Training models...")
        
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
        }
        
        for name, model in models_to_train.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate models"""
        print("\n📏 Evaluating models...")
        print("="*90)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'AUC':<10}")
        print("="*90)
        
        results = {}
        best_auc = 0
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'AUC': auc}
            
            print(f"{name:<25} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<10.4f} {auc:<10.4f}")
            
            if auc > best_auc:
                best_auc = auc
                self.best_model_name = name
                self.best_model = model
        
        print("="*90)
        print(f"\n🏆 Best Model: {self.best_model_name} (AUC: {best_auc:.4f})")
        
        results_df = pd.DataFrame(results).T
        results_df.to_csv(self.results_dir / 'classification_results.csv')
        
        return results
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves"""
        print("\n📈 Creating ROC curves...")
        
        plt.figure(figsize=(10, 6))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Admission Probability Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'roc_curves.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved")
    
    def plot_confusion_matrix(self, X_test, y_test):
        """Plot confusion matrix"""
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Admitted', 'Admitted'],
                   yticklabels=['Not Admitted', 'Admitted'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def plot_calibration(self, X_test, y_test):
        """Plot calibration"""
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        fraction_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_pred, fraction_pos, 'o-', label=self.best_model_name)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'calibration_plot.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def feature_importance(self):
        """Feature importance"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Features:")
            print(feature_df.head(10).to_string(index=False))
            
            feature_df.to_csv(self.results_dir / 'classification_feature_importance.csv', index=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_df.head(10), x='Importance', y='Feature')
            plt.title(f'Top 10 Features - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'classification_feature_importance.png', dpi=100)
            plt.close()
    
    def save_model(self):
        """Save model"""
        model_file = self.output_dir / 'admission_probability_model.pkl'
        joblib.dump({
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_columns': self.feature_columns
        }, model_file)
        print(f"\n💾 Model saved to {model_file}")
    
    def run_pipeline(self):
        """Execute pipeline"""
        print("="*60)
        print("🤖 ADMISSION PROBABILITY MODEL (FIXED - NO LEAKAGE)")
        print("="*60)
        
        self.load_data()
        self.create_realistic_training_data()
        self.prepare_features()
        X_train, X_test, y_train, y_test = self.split_data()
        self.train_models(X_train, y_train)
        results = self.evaluate_models(X_test, y_test)
        self.plot_roc_curves(X_test, y_test)
        self.plot_confusion_matrix(X_test, y_test)
        self.plot_calibration(X_test, y_test)
        self.feature_importance()
        self.save_model()
        
        print("\n" + "="*60)
        print("✅ MODEL 2 COMPLETE!")
        print("="*60)
        
        return results


if __name__ == "__main__":
    model = AdmissionProbabilityModelFixed()
    model.run_pipeline()
