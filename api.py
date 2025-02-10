from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved model
with open('modelforest.pkl', 'rb') as f:
    model = pickle.load(f)

# Load class names from dataset.csv
df = pd.read_csv('dataset.csv')
class_names = df['label'].unique().tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the request
        features = [
            float(request.form['rainfall']),
            float(request.form['temp']),
            float(request.form['weather']),
            float(request.form['ph']),
            float(request.form['hum'])
        ]
        
        # Make prediction
        probs = model.predict_proba([features])[0]
        
        # Create prediction results
        results = [
            {'crop': crop, 'probability': round(prob * 100, 2)}
            for crop, prob in zip(class_names, probs)
        ]
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        # Get the recommended crop (highest probability)
        recommended_crop = results[0]['crop']
        
        return jsonify({
            'success': True,
            'recommended_crop': recommended_crop,
            'probability': results[0]['probability']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Print loaded class names for verification
    print("Loaded classes:", class_names)
    app.run(debug=True)
