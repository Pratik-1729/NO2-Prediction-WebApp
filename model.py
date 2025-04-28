import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LassoCV, RidgeCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin, clone
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class StackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_predictions = None

    def fit(self, X, y):
        # Train base models
        self.base_predictions = np.column_stack([
            cross_val_predict(clone(model), X, y, cv=5)
            for model in self.base_models
        ])
        
        # Train base models on full dataset
        for model in self.base_models:
            model.fit(X, y)
            
        # Train meta model
        self.meta_model.fit(self.base_predictions, y)
        return self

    def predict(self, X):
        # Make predictions with base models
        predictions = np.column_stack([
            model.predict(X)
            for model in self.base_models
        ])
        
        # Make final prediction
        return self.meta_model.predict(predictions)

# Load processed data
X_train = np.load('processed_data/X_train.npy')
X_test = np.load('processed_data/X_test.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')

# Load selected features
with open('processed_data/selected_features.txt', 'r') as f:
    selected_features = f.read().splitlines()

print("Training multiple models...")

# 1. Random Forest with GridSearch
rf_params = {
    'n_estimators': [200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print("\nBest Random Forest Parameters:", rf_grid.best_params_)
print("Random Forest CV Score:", rf_grid.best_score_)

# 2. XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=42)
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='r2', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
print("\nBest XGBoost Parameters:", xgb_grid.best_params_)
print("XGBoost CV Score:", xgb_grid.best_score_)

# 3. LightGBM
lgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [31, 63, 127],
    'subsample': [0.8, 0.9, 1.0]
}

lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=5, scoring='r2', n_jobs=-1)
lgb_grid.fit(X_train, y_train)
print("\nBest LightGBM Parameters:", lgb_grid.best_params_)
print("LightGBM CV Score:", lgb_grid.best_score_)

# 4. Gradient Boosting
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0]
}

gb = GradientBoostingRegressor(random_state=42)
gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='r2', n_jobs=-1)
gb_grid.fit(X_train, y_train)
print("\nBest Gradient Boosting Parameters:", gb_grid.best_params_)
print("Gradient Boosting CV Score:", gb_grid.best_score_)

# Create ensemble using stacking
base_models = [
    rf_grid.best_estimator_,
    xgb_grid.best_estimator_,
    lgb_grid.best_estimator_,
    gb_grid.best_estimator_
]

meta_model = RidgeCV()
stacking_model = StackingRegressor(base_models, meta_model)
stacking_model.fit(X_train, y_train)

# Make predictions with all models
predictions = {
    'Random Forest': rf_grid.predict(X_test),
    'XGBoost': xgb_grid.predict(X_test),
    'LightGBM': lgb_grid.predict(X_test),
    'Gradient Boosting': gb_grid.predict(X_test),
    'Stacking': stacking_model.predict(X_test)
}

# Evaluate all models
print("\nTest Set Metrics:")
for name, pred in predictions.items():
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    print(f"\n{name}:")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")

# Select the best model based on test set performance
best_r2 = -1
best_model_name = None
best_predictions = None

for name, pred in predictions.items():
    r2 = r2_score(y_test, pred)
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_predictions = pred

print(f"\nBest Model: {best_model_name}")
print(f"Best R2 Score: {best_r2:.4f}")

# Plot actual vs predicted values for best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual NO2 Concentration')
plt.ylabel('Predicted NO2 Concentration')
plt.title(f'Actual vs Predicted NO2 Concentration ({best_model_name})')
plt.savefig('actual_vs_predicted.png')
plt.close()

# Feature importance plot (using Random Forest as reference)
feature_importance = rf_grid.best_estimator_.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 8))
plt.barh(pos, feature_importance[sorted_idx])
plt.yticks(pos, np.array(selected_features)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save the best model
if best_model_name == 'Stacking':
    joblib.dump(stacking_model, 'model/no2_predictor.pkl')
else:
    model_dict = {
        'Random Forest': rf_grid.best_estimator_,
        'XGBoost': xgb_grid.best_estimator_,
        'LightGBM': lgb_grid.best_estimator_,
        'Gradient Boosting': gb_grid.best_estimator_
    }
    joblib.dump(model_dict[best_model_name], 'model/no2_predictor.pkl')

# Additional analysis
print("\nModel Correlation Analysis:")
pred_df = pd.DataFrame(predictions)
correlation_matrix = pred_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Model Predictions Correlation Matrix')
plt.savefig('model_correlations.png')
plt.close()

# Print feature importance for all models that support it
print("\nFeature Importance Analysis:")
for name, model in [('Random Forest', rf_grid.best_estimator_), 
                   ('XGBoost', xgb_grid.best_estimator_),
                   ('LightGBM', lgb_grid.best_estimator_)]:
    if hasattr(model, 'feature_importances_'):
        print(f"\n{name} Feature Importance:")
        for feature, importance in zip(selected_features, model.feature_importances_):
            print(f"{feature}: {importance:.4f}") 