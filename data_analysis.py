import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import missingno as msno
from scipy import stats, fft
import warnings
warnings.filterwarnings('ignore')

def create_fourier_features(data, date_column, n_harmonics=4):
    """Create Fourier features for seasonality"""
    # Convert dates to days since the start
    days = (data[date_column] - data[date_column].min()).dt.total_seconds() / (24 * 60 * 60)
    fourier_features = pd.DataFrame(index=data.index)
    
    # Base frequency for yearly seasonality (365.25 days)
    base_freq = 2 * np.pi / 365.25
    
    for n in range(1, n_harmonics + 1):
        fourier_features[f'sin_year_{n}'] = np.sin(n * base_freq * days)
        fourier_features[f'cos_year_{n}'] = np.cos(n * base_freq * days)
        # Add monthly seasonality
        fourier_features[f'sin_month_{n}'] = np.sin(n * 12 * base_freq * days)
        fourier_features[f'cos_month_{n}'] = np.cos(n * 12 * base_freq * days)
    
    return fourier_features

def create_weather_patterns(data, n_clusters=5):
    """Create weather pattern classifications"""
    weather_features = ['temperature_K', 'relative_humidity_percent', 'wind_speed_m_s', 'solar_rad_J_m2']
    
    # Create a copy of the data for clustering
    cluster_data = data[weather_features].copy()
    
    # Ensure we have a proper DatetimeIndex
    if not isinstance(cluster_data.index, pd.DatetimeIndex):
        cluster_data.index = pd.to_datetime(cluster_data.index)
    
    # Handle missing values in weather features
    # First try interpolation for time series data
    for feature in weather_features:
        cluster_data[feature] = cluster_data[feature].interpolate(method='time')
    
    # Fill any remaining missing values with median
    imputer = SimpleImputer(strategy='median')
    cluster_data = pd.DataFrame(
        imputer.fit_transform(cluster_data),
        columns=cluster_data.columns,
        index=cluster_data.index
    )
    
    # Standardize the features
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    weather_patterns = kmeans.fit_predict(cluster_data_scaled)
    
    return pd.get_dummies(weather_patterns, prefix='weather_pattern')

# Load the dataset
df = pd.read_csv('data/NO2_NDVI_Weather_Combined.csv')

# Convert date to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print("\nInitial index type:", type(df.index))
print("Is DatetimeIndex:", isinstance(df.index, pd.DatetimeIndex))

# Basic EDA
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

# Handle missing values first
print("\nHandling missing values...")
# Use interpolation for time series data
df['no2_mol_m2'] = df['no2_mol_m2'].interpolate(method='time')
df['ndvi'] = df['ndvi'].interpolate(method='time')

# Fill any remaining missing values with median
df_imputed = df.copy()
for column in df_imputed.columns:
    if df_imputed[column].isnull().any():
        df_imputed[column] = df_imputed[column].fillna(df_imputed[column].median())

print("\nAfter imputation index type:", type(df_imputed.index))
print("Is DatetimeIndex:", isinstance(df_imputed.index, pd.DatetimeIndex))

# Advanced Feature Engineering

# 1. Fourier features for seasonality
fourier_features = create_fourier_features(df_imputed.reset_index(), 'date')
df_imputed = pd.concat([df_imputed, fourier_features], axis=1)

# 2. Weather patterns
weather_patterns = create_weather_patterns(df_imputed)
df_imputed = pd.concat([df_imputed, weather_patterns], axis=1)

# 3. Rolling statistics for weather features
windows = [3, 7, 14]  # Adding 14-day window
for window in windows:
    df_imputed[f'temp_rolling_mean_{window}d'] = df_imputed['temperature_K'].rolling(window=window).mean()
    df_imputed[f'temp_rolling_std_{window}d'] = df_imputed['temperature_K'].rolling(window=window).std()
    df_imputed[f'humidity_rolling_mean_{window}d'] = df_imputed['relative_humidity_percent'].rolling(window=window).mean()
    df_imputed[f'humidity_rolling_std_{window}d'] = df_imputed['relative_humidity_percent'].rolling(window=window).std()
    df_imputed[f'wind_rolling_mean_{window}d'] = df_imputed['wind_speed_m_s'].rolling(window=window).mean()
    df_imputed[f'wind_rolling_std_{window}d'] = df_imputed['wind_speed_m_s'].rolling(window=window).std()

# 4. Advanced weather indices
df_imputed['temp_celsius'] = df_imputed['temperature_K'] - 273.15
df_imputed['THI'] = (0.8 * df_imputed['temp_celsius']) + (df_imputed['relative_humidity_percent'] / 100) * (df_imputed['temp_celsius'] - 14.4) + 46.4
df_imputed['wind_chill'] = 13.12 + 0.6215 * df_imputed['temp_celsius'] - 11.37 * (df_imputed['wind_speed_m_s'] * 3.6)**0.16 + 0.3965 * df_imputed['temp_celsius'] * (df_imputed['wind_speed_m_s'] * 3.6)**0.16
df_imputed['heat_index'] = -8.784695 + 1.61139411 * df_imputed['temp_celsius'] + 2.338549 * df_imputed['relative_humidity_percent'] - 0.14611605 * df_imputed['temp_celsius'] * df_imputed['relative_humidity_percent']

# 5. Wind-related features
df_imputed['wind_direction_factor'] = np.sin(df_imputed['wind_speed_m_s']) * np.cos(df_imputed['wind_speed_m_s'])
df_imputed['wind_speed_squared'] = df_imputed['wind_speed_m_s'] ** 2
df_imputed['wind_gust_factor'] = df_imputed['wind_speed_squared'] * df_imputed['relative_humidity_percent'] / 100

# 6. Temporal features
print("\nCreating temporal features...")
print("Current index type:", type(df_imputed.index))
print("Index values:", df_imputed.index[:5])

# Convert index to datetime if it's not already
if not isinstance(df_imputed.index, pd.DatetimeIndex):
    df_imputed.index = pd.to_datetime(df_imputed.index)

# Create temporal features using pandas datetime methods
df_imputed['year'] = df_imputed.index.map(lambda x: x.year)
df_imputed['month'] = df_imputed.index.map(lambda x: x.month)
df_imputed['day'] = df_imputed.index.map(lambda x: x.day)
df_imputed['day_of_week'] = df_imputed.index.map(lambda x: x.dayofweek)
df_imputed['season'] = df_imputed.index.map(lambda x: (x.month % 12) // 3 + 1)
df_imputed['is_weekend'] = df_imputed.index.map(lambda x: x.dayofweek in [5, 6]).astype(int)
df_imputed['day_of_year'] = df_imputed.index.map(lambda x: x.dayofyear)
df_imputed['week_of_year'] = df_imputed.index.map(lambda x: x.isocalendar().week)
df_imputed['is_holiday'] = df_imputed.index.map(lambda x: (x.month == 12 and x.day == 25) or (x.month == 1 and x.day == 1)).astype(int)

# 7. Advanced interaction features
df_imputed['temp_humidity_interaction'] = df_imputed['temperature_K'] * df_imputed['relative_humidity_percent']
df_imputed['temp_wind_interaction'] = df_imputed['temperature_K'] * df_imputed['wind_speed_m_s']
df_imputed['humidity_wind_interaction'] = df_imputed['relative_humidity_percent'] * df_imputed['wind_speed_m_s']
df_imputed['solar_temp_interaction'] = df_imputed['solar_rad_J_m2'] * df_imputed['temperature_K']
df_imputed['ndvi_temp_interaction'] = df_imputed['ndvi'] * df_imputed['temperature_K']

# 8. Time series features
for lag in range(1, 8):  # Increasing lag features
    df_imputed[f'no2_lag_{lag}'] = df_imputed['no2_mol_m2'].shift(lag)

# Rolling statistics for NO2
for window in windows:
    df_imputed[f'no2_rolling_mean_{window}d'] = df_imputed['no2_mol_m2'].rolling(window=window).mean()
    df_imputed[f'no2_rolling_std_{window}d'] = df_imputed['no2_mol_m2'].rolling(window=window).std()
    df_imputed[f'no2_rolling_max_{window}d'] = df_imputed['no2_mol_m2'].rolling(window=window).max()
    df_imputed[f'no2_rolling_min_{window}d'] = df_imputed['no2_mol_m2'].rolling(window=window).min()

# 9. Rate of change features
df_imputed['no2_rate_of_change'] = df_imputed['no2_mol_m2'].diff()
df_imputed['temp_rate_of_change'] = df_imputed['temperature_K'].diff()
df_imputed['humidity_rate_of_change'] = df_imputed['relative_humidity_percent'].diff()

# Fill any NaN values created by rolling windows and lag features
df_imputed = df_imputed.fillna(method='bfill').fillna(method='ffill')

# Prepare features and target
features = [col for col in df_imputed.columns if col not in ['no2_mol_m2', 'temp_celsius']]
target = 'no2_mol_m2'

X = df_imputed[features]
y = df_imputed[target]

# Feature selection using mutual information
selector = SelectKBest(score_func=mutual_info_regression, k=25)  # Increasing selected features
X_selected = selector.fit_transform(X, y)
selected_features = [features[i] for i in selector.get_support(indices=True)]
print("\nSelected Features:", selected_features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Scale the features using RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
np.save('processed_data/X_train.npy', X_train_scaled)
np.save('processed_data/X_test.npy', X_test_scaled)
np.save('processed_data/y_train.npy', y_train)
np.save('processed_data/y_test.npy', y_test)

# Save the scaler and feature selector
joblib.dump(scaler, 'processed_data/scaler.pkl')
joblib.dump(selector, 'processed_data/feature_selector.pkl')

# Save the list of selected features
with open('processed_data/selected_features.txt', 'w') as f:
    f.write('\n'.join(selected_features)) 