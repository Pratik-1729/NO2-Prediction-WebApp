# NO2 Prediction Web Application

This web application predicts NO2 (Nitrogen Dioxide) concentration based on weather data and location. It uses machine learning to forecast NO2 levels and displays current weather conditions.

## Features

- Real-time NO2 concentration prediction
- Current weather data integration
- Location-based predictions
- Interactive web interface
- Responsive design

## Prerequisites

- Python 3.8 or higher
- OpenWeather API key

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenWeather API key:
```
OPENWEATHER_API_KEY=your_api_key_here
```

## Project Structure

```
NO2_PREDICTION/
├── data/
│   └── NO2_NDVI_Weather_Combined.csv
├── processed_data/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   └── scaler.pkl
├── model/
│   └── no2_predictor.pkl
├── templates/
│   └── index.html
├── app.py
├── data_analysis.py
├── model.py
├── requirements.txt
└── README.md
```

## Usage

1. Run the data analysis script to process the data:
```bash
python data_analysis.py
```

2. Train the model:
```bash
python model.py
```

3. Start the web application:
```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## How to Use

1. Click "Use Current Location" to automatically detect your location, or
2. Enter latitude and longitude manually
3. Click "Predict NO2" to get the prediction
4. View the predicted NO2 concentration and current weather conditions

## Model Details

The prediction model uses a Random Forest Regressor trained on historical NO2 and weather data. Features include:
- Precipitation
- Relative humidity
- Solar radiation
- Temperature
- Wind speed
- NDVI (Normalized Difference Vegetation Index)
- Temporal features (year, month, day, day of week)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 