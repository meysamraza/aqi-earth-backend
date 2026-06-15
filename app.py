from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import requests
import time

app = Flask(__name__)
CORS(app)

# =====================
# Load Models
# =====================
knn_final = joblib.load('model/knn_model.pkl')
nb        = joblib.load('model/nb_model.pkl')
scaler    = joblib.load('model/scaler.pkl')
encoder   = joblib.load('model/encoder.pkl')

# =====================
# Load precomputed city averages
print("Loading city data...")
city_avg = pd.read_csv("city_avg.csv")

# =====================
# Fetch City Coordinates
# =====================
def get_coords(city, state):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={city},{state},USA&format=json&limit=1"
        r = requests.get(url, headers={'User-Agent': 'aqi-app'}, timeout=5)
        data = r.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except:
        pass
    return None, None

print("Fetching city coordinates (one time)...")
lats, lons = [], []
for _, row in city_avg.iterrows():
    lat, lon = get_coords(row['City'], row['State'])
    lats.append(lat)
    lons.append(lon)
    time.sleep(1)

city_avg['lat'] = lats
city_avg['lon'] = lons
city_avg = city_avg.dropna(subset=['lat', 'lon']).reset_index(drop=True)

coords = city_avg[['lat', 'lon']].values
tree = KDTree(coords)
print(f"Ready! {len(city_avg)} cities loaded.")

# =====================
# Routes
# =====================
@app.route('/')
def home():
    return jsonify({'status': 'AQI API running'})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = np.array([[
        data['O3 Mean'],
        data['CO Mean'],
        data['SO2 Mean'],
        data['NO2 Mean'],
        data['O3 AQI'],
        data['CO AQI'],
        data['SO2 AQI'],
        data['NO2 AQI']
    ]])

    features_scaled = scaler.transform(features)
    model_choice = data.get('model', 'knn')

    if model_choice == 'nb':
        pred = nb.predict(features_scaled)
    else:
        pred = knn_final.predict(features_scaled)

    category = encoder.inverse_transform(pred)[0]

    return jsonify({
        'category': category,
        'model': model_choice
    })


@app.route('/location', methods=['GET'])
def location():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
    except:
        return jsonify({'error': 'Invalid lat/lon'}), 400

    dist, idx = tree.query([lat, lon])
    nearest = city_avg.iloc[idx]

    features = np.array([[
        nearest['O3 Mean'],
        nearest['CO Mean'],
        nearest['SO2 Mean'],
        nearest['NO2 Mean'],
        nearest['O3 AQI'],
        nearest['CO AQI'],
        nearest['SO2 AQI'],
        nearest['NO2 AQI']
    ]])

    features_scaled = scaler.transform(features)
    pred = knn_final.predict(features_scaled)
    category = encoder.inverse_transform(pred)[0]

    return jsonify({
        'city':     nearest['City'],
        'state':    nearest['State'],
        'category': category,
        'aqi_values': {
            'O3 AQI':  round(float(nearest['O3 AQI']), 1),
            'CO AQI':  round(float(nearest['CO AQI']), 1),
            'SO2 AQI': round(float(nearest['SO2 AQI']), 1),
            'NO2 AQI': round(float(nearest['NO2 AQI']), 1)
        }
    })


if __name__ == '__main__':
    app.run(debug=True)