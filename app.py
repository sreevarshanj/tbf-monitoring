from flask import Flask, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the KNN model directly from 'secret.pkl'
model = joblib.load('secret.pkl')

# Load the new CSV file with updated parameters
df = pd.read_csv('Dataset.csv')

# Define threshold values for each parameter (adjust as necessary)
THRESHOLDS = {
    'rpm': 5000,           # Example threshold
    'temp': 40,           # Example threshold in °C
    'humidity': 80,        # Example threshold in %
    'gasdifference': 130    # Example threshold (units may vary)
}

# Route to serve the HTML interface
@app.route('/')
def index():
    return render_template('index.html')

# Route to get data from the CSV and return parameters and anomaly prediction
@app.route('/start/<int:index>', methods=['GET'])
def start_monitoring(index):
    # Fetch the row at the given index from the CSV
    if index >= len(df):
        return jsonify({'error': 'Index out of range'})

    row = df.iloc[index]
    try:
        input_data = pd.DataFrame([row[['rpm', 'temp', 'humidity', 'gasdifference']]])
    except KeyError as e:
        return jsonify({'error': f'KeyError: {str(e)}'})

    # Predict anomaly using the ML model
    try:
        anomaly = model.predict(input_data)[0]  # Predict using the model
        print("Prediction:", anomaly)
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

    # Check which parameter(s) might have caused the anomaly based on thresholds
    anomaly_params = []
    if row['rpm'] < THRESHOLDS['rpm'] or row['rpm'] > 8000:
        anomaly_params.append('RPM')
    if row['temp'] > THRESHOLDS['temp']:
        anomaly_params.append('Temperature')
    if row['humidity'] > THRESHOLDS['humidity'] or row['humidity'] < 60:
        anomaly_params.append('Humidity')
    if row['gasdifference'] < THRESHOLDS['gasdifference']:
        anomaly_params.append('Gas Difference')

    anomaly_info = 'None' if not anomaly_params else ', '.join(anomaly_params)

    # Convert all values to native Python types before returning them as JSON
    return jsonify({
        'time': str(row.get('timestamp', 'No data')),
        'rpm': float(row.get('rpm', 0)),
        'temp': float(row.get('temp', 0)),
        'humidity': float(row.get('humidity', 0)),
        'gasdifference': float(row.get('gasdifference', 0)),
        'anomaly': int(anomaly),  # Convert to native Python int
        'anomaly_params': anomaly_info  # Send the parameters that caused the anomaly
    })

if __name__ == '__main__':
    app.run(debug=True)
