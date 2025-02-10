from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
with open('modelforest2.pkl', 'rb') as f:
    random_forest = pickle.load(f)

# Load the dataset to get the class names
dataset = pd.read_csv('dataset.csv')
class_names = dataset['label'].unique()

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request Content-Type is application/json
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    
    # Get the input data from the request
    data = request.get_json()
    
    # Check if 'features' key exists in the JSON data
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' key in JSON data"}), 400
    
    # Convert the input data to a numpy array
    features = np.array(data['features']).reshape(1, -1)
    
    # Predict probabilities for each class
    rf_probs = random_forest.predict_proba(features)[0]
    
    # Get the predicted class
    predicted_class = class_names[np.argmax(rf_probs)]
    
    # Create a dictionary of probabilities
    probabilities = {class_name: float(prob) for class_name, prob in zip(class_names, rf_probs)}
    
    # Get the highest probability and its corresponding class
    highest_prob_class = max(probabilities, key=probabilities.get)
    highest_prob_value = probabilities[highest_prob_class]
    
    # Prepare the response
    response = {
        'recommended_crop': predicted_class,
        'probability' : highest_prob_value * 100,
       
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
