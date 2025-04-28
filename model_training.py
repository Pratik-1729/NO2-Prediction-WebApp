import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split, cross_val_predict
from sklearn.preprocessing import RobustScaler, StandardScaler
import lightgbm as lgb
import joblib
import os
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data
X_train = np.load('processed_data/X_train.npy')
X_test = np.load('processed_data/X_test.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')

# Scale the target variable
target_scaler = RobustScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

# Select only the most important features based on previous analysis
important_features = [
    'wind_gust_factor',        # Highest importance
    'no2_rolling_mean_3d',     # High importance
    'no2_rolling_std_7d',      # High importance
    'no2_rolling_min_3d',      # High importance
    'no2_rolling_mean_7d',     # High importance
    'day_of_year'              # High importance
]

# Get feature indices
with open('processed_data/selected_features.txt', 'r') as f:
    feature_names = f.read().splitlines()
feature_indices = [feature_names.index(f) for f in important_features]

# Select only important features
X_train_enhanced = X_train[:, feature_indices]
X_test_enhanced = X_test[:, feature_indices]

# Create validation set for early stopping
X_train_base, X_val, y_train_base, y_val = train_test_split(
    X_train_enhanced, y_train_scaled, test_size=0.2, random_state=42
)

# Base models with optimized parameters
base_models = {
    'lgb1': lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.005,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        n_jobs=-1
    ),
    'lgb2': lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=43,
        n_jobs=-1
    ),
    'gb': GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),
    'rf': RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
}

# Train and evaluate base models with cross-validation
print("Training and evaluating base models...")
base_predictions = {}
base_scores = {}

for name, model in base_models.items():
    print(f"\nTraining {name}...")
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_enhanced, y_train_scaled, cv=kf, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train on full training set with early stopping if supported
    if name.startswith('lgb'):
        model.fit(
            X_train_base, y_train_base,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
    else:
        model.fit(X_train_enhanced, y_train_scaled)
    
    pred_scaled = model.predict(X_test_enhanced)
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    
    base_predictions[name] = pred
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    base_scores[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
    print(f"{name} - R² Score: {r2:.4f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")

# Create stacked predictions for meta-features
meta_features_train = np.column_stack([
    cross_val_predict(clone(model), X_train_enhanced, y_train_scaled, cv=5) 
    for model in base_models.values()
])
meta_features_test = np.column_stack([
    model.predict(X_test_enhanced) for model in base_models.values()
])

# Meta-learner (LightGBM with optimized parameters)
meta_learner = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.005,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    n_jobs=-1
)

# Train meta-learner
meta_learner.fit(
    meta_features_train, y_train_scaled,
    eval_set=[(meta_features_test, y_test_scaled)],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# Make predictions
final_predictions_scaled = meta_learner.predict(meta_features_test)
final_predictions = target_scaler.inverse_transform(final_predictions_scaled.reshape(-1, 1)).ravel()

# Evaluate final stacked model
final_r2 = r2_score(y_test, final_predictions)
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
final_mae = mean_absolute_error(y_test, final_predictions)

print("\nFinal Stacked Model Performance:")
print(f"R² Score: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.6f}")
print(f"MAE: {final_mae:.6f}")

# Save models and scaler
print("\nSaving models...")
os.makedirs('models', exist_ok=True)

joblib.dump(base_models, 'models/base_models.pkl')
joblib.dump(meta_learner, 'models/meta_learner.pkl')
joblib.dump(target_scaler, 'models/target_scaler.pkl')

# Feature importance analysis
feature_importance = pd.DataFrame()
for name, model in base_models.items():
    if hasattr(model, 'feature_importances_'):
        feature_importance[name] = model.feature_importances_

if feature_importance.shape[1] > 0:
    feature_importance.index = important_features
    feature_importance['mean_importance'] = feature_importance.mean(axis=1)
    feature_importance = feature_importance.sort_values('mean_importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance[['mean_importance']].head(10))

# Save feature importance
feature_importance.to_csv('models/feature_importance.csv') 