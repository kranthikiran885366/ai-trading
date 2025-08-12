import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')
import logging
from typing import Dict, List, Tuple, Optional, Any
import optuna
from optuna.samplers import TPESampler
import time

class AdvancedEnsembleModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.is_trained = False
        self.feature_importance = None
        self.model_weights = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize base models
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Initialize all base models with optimized parameters"""
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                max_features='sqrt',
                random_state=self.random_state
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            
            'catboost': cb.CatBoostClassifier(
                iterations=200,
                learning_rate=0.1,
                depth=8,
                l2_leaf_reg=3,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                random_seed=self.random_state,
                verbose=False
            ),
            
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            
            'logistic_regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                max_iter=1000,
                random_state=self.random_state
            ),
            
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            ),
            
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                algorithm='auto',
                n_jobs=-1
            ),
            
            'naive_bayes': GaussianNB(),
            
            'ada_boost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                algorithm='SAMME.R',
                random_state=self.random_state
            )
        }
    
    def optimize_hyperparameters(self, X, y, model_name, n_trials=100):
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestClassifier(**params, random_state=self.random_state, n_jobs=-1)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
                model = xgb.XGBClassifier(**params, random_state=self.random_state, n_jobs=-1, eval_metric='mlogloss')
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
                model = lgb.LGBMClassifier(**params, random_state=self.random_state, n_jobs=-1, verbose=-1)
                
            else:
                return 0  # Skip optimization for other models
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params
    
    def train_base_models(self, X, y, optimize_hyperparams=False):
        """Train all base models"""
        self.logger.info("Training base models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        self.feature_selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        model_scores = {}
        
        for name, model in self.base_models.items():
            self.logger.info(f"Training {name}...")
            
            try:
                # Optimize hyperparameters if requested
                if optimize_hyperparams and name in ['random_forest', 'xgboost', 'lightgbm']:
                    self.logger.info(f"Optimizing hyperparameters for {name}...")
                    best_params = self.optimize_hyperparameters(X_selected, y, name)
                    
                    # Update model with best parameters
                    if name == 'random_forest':
                        self.base_models[name] = RandomForestClassifier(
                            **best_params, random_state=self.random_state, n_jobs=-1
                        )
                    elif name == 'xgboost':
                        self.base_models[name] = xgb.XGBClassifier(
                            **best_params, random_state=self.random_state, n_jobs=-1, eval_metric='mlogloss'
                        )
                    elif name == 'lightgbm':
                        self.base_models[name] = lgb.LGBMClassifier(
                            **best_params, random_state=self.random_state, n_jobs=-1, verbose=-1
                        )
                
                # Cross-validation
                cv_scores = cross_val_score(
                    self.base_models[name], X_selected, y, cv=tscv, scoring='accuracy', n_jobs=-1
                )
                model_scores[name] = cv_scores.mean()
                
                # Train on full dataset
                self.base_models[name].fit(X_selected, y)
                
                self.logger.info(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                # Remove failed model
                del self.base_models[name]
        
        # Calculate model weights based on performance
        total_score = sum(model_scores.values())
        self.model_weights = {name: score / total_score for name, score in model_scores.items()}
        
        self.logger.info(f"Model weights: {self.model_weights}")
        
        # Calculate feature importance
        self._calculate_feature_importance(X_selected)
        
        self.logger.info("Base models training completed")
    
    def _calculate_feature_importance(self, X):
        """Calculate ensemble feature importance"""
        feature_importances = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importances[name] = np.abs(model.coef_[0])
        
        if feature_importances:
            # Average feature importance across models
            all_importances = np.array(list(feature_importances.values()))
            self.feature_importance = np.mean(all_importances, axis=0)
    
    def create_ensemble(self, X, y, meta_model_type='logistic'):
        """Create ensemble using stacking"""
        self.logger.info("Creating ensemble with stacking...")
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Generate meta-features using cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = np.zeros((X_selected.shape[0], len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_selected)):
            X_train_fold, X_val_fold = X_selected[train_idx], X_selected[val_idx]
            y_train_fold = y[train_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                # Clone and train model on fold
                fold_model = type(model)(**model.get_params())
                fold_model.fit(X_train_fold, y_train_fold)
                
                # Predict on validation set
                if hasattr(fold_model, 'predict_proba'):
                    pred_proba = fold_model.predict_proba(X_val_fold)
                    meta_features[val_idx, i] = pred_proba[:, 1] if pred_proba.shape[1] == 2 else np.max(pred_proba, axis=1)
                else:
                    meta_features[val_idx, i] = fold_model.predict(X_val_fold)
        
        # Train meta-model
        if meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(random_state=self.random_state)
        elif meta_model_type == 'random_forest':
            self.meta_model = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
        elif meta_model_type == 'xgboost':
            self.meta_model = xgb.XGBClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1, eval_metric='mlogloss'
            )
        
        self.meta_model.fit(meta_features, y)
        
        self.is_trained = True
        self.logger.info("Ensemble creation completed")
    
    def predict(self, X):
        """Make predictions using ensemble"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained yet")
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Get base model predictions
        base_predictions = np.zeros((X_selected.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X_selected)
                base_predictions[:, i] = pred_proba[:, 1] if pred_proba.shape[1] == 2 else np.max(pred_proba, axis=1)
            else:
                base_predictions[:, i] = model.predict(X_selected)
        
        # Meta-model prediction
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions
    
    def predict_proba(self, X):
        """Make probability predictions using ensemble"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained yet")
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Get base model predictions
        base_predictions = np.zeros((X_selected.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X_selected)
                base_predictions[:, i] = pred_proba[:, 1] if pred_proba.shape[1] == 2 else np.max(pred_proba, axis=1)
            else:
                base_predictions[:, i] = model.predict(X_selected)
        
        # Meta-model probability prediction
        if hasattr(self.meta_model, 'predict_proba'):
            final_probabilities = self.meta_model.predict_proba(base_predictions)
        else:
            # Convert predictions to probabilities
            predictions = self.meta_model.predict(base_predictions)
            n_classes = len(np.unique(predictions))
            final_probabilities = np.eye(n_classes)[predictions]
        
        return final_probabilities
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.feature_importance is not None:
            selected_features = self.feature_selector.get_support()
            importance_dict = {}
            
            for i, is_selected in enumerate(selected_features):
                if is_selected:
                    feature_idx = np.sum(selected_features[:i+1]) - 1
                    if feature_idx < len(self.feature_importance):
                        importance_dict[f'feature_{i}'] = self.feature_importance[feature_idx]
            
            return importance_dict
        return None
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def save_ensemble(self, filepath):
        """Save ensemble model"""
        ensemble_data = {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        
        joblib.dump(ensemble_data, filepath)
        self.logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath):
        """Load ensemble model"""
        ensemble_data = joblib.load(filepath)
        
        self.base_models = ensemble_data['base_models']
        self.meta_model = ensemble_data['meta_model']
        self.scaler = ensemble_data['scaler']
        self.feature_selector = ensemble_data['feature_selector']
        self.model_weights = ensemble_data['model_weights']
        self.feature_importance = ensemble_data['feature_importance']
        self.is_trained = ensemble_data['is_trained']
        
        self.logger.info(f"Ensemble loaded from {filepath}")

class AutoMLPipeline:
    """Automated Machine Learning Pipeline"""
    
    def __init__(self, time_budget=3600, random_state=42):
        self.time_budget = time_budget
        self.random_state = random_state
        self.best_model = None
        self.best_score = 0
        self.model_results = {}
        self.logger = logging.getLogger(__name__)
    
    def run_automl(self, X, y, metric='accuracy'):
        """Run AutoML pipeline"""
        self.logger.info("Starting AutoML pipeline...")
        
        start_time = time.time()
        
        # Model configurations to try
        model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier,
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9],
                    'min_child_weight': [1, 3, 5]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier,
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9],
                    'min_child_samples': [5, 10, 20]
                }
            }
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for model_name, config in model_configs.items():
            if time.time() - start_time > self.time_budget:
                break
            
            self.logger.info(f"Optimizing {model_name}...")
            
            try:
                # Grid search with time series CV
                grid_search = GridSearchCV(
                    config['model'](random_state=self.random_state),
                    config['param_grid'],
                    cv=tscv,
                    scoring=metric,
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X, y)
                
                score = grid_search.best_score_
                self.model_results[model_name] = {
                    'model': grid_search.best_estimator_,
                    'score': score,
                    'params': grid_search.best_params_
                }
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = grid_search.best_estimator_
                
                self.logger.info(f"{model_name} best score: {score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error optimizing {model_name}: {e}")
        
        self.logger.info(f"AutoML completed. Best score: {self.best_score:.4f}")
        return self.best_model, self.best_score
    
    def get_model_comparison(self):
        """Get comparison of all models"""
        comparison = pd.DataFrame([
            {
                'model': name,
                'score': results['score'],
                'params': str(results['params'])
            }
            for name, results in self.model_results.items()
        ]).sort_values('score', ascending=False)
        
        return comparison

def main():
    """Test the ensemble model"""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test ensemble model
    ensemble = AdvancedEnsembleModel()
    
    print("Training base models...")
    ensemble.train_base_models(X_train, y_train)
    
    print("Creating ensemble...")
    ensemble.create_ensemble(X_train, y_train)
    
    print("Evaluating ensemble...")
    results = ensemble.evaluate_ensemble(X_test, y_test)
    
    # Test AutoML
    print("\nTesting AutoML...")
    automl = AutoMLPipeline(time_budget=300)  # 5 minutes
    best_model, best_score = automl.run_automl(X_train, y_train)
    
    print(f"AutoML best score: {best_score:.4f}")
    print("Model comparison:")
    print(automl.get_model_comparison())

if __name__ == "__main__":
    main()
