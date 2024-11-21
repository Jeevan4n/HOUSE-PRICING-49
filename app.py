import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the regression model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Load the dataset to extract feature names
df = pd.read_csv('manu.csv')  # Replace with your actual dataset file
features = df.columns.tolist()  # Extract the column names (features)

# Home route to render the HTML page
@app.route('/')
def home():
    return render_template('home.html', features=features)

# API endpoint to predict using JSON input
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']  # Input data sent as JSON
    print(data)  # For debugging: Print the received data
    print(np.array(list(data.values())).reshape(1, -1))  # Reshape for prediction
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))  # Scaling the input
    output = regmodel.predict(new_data)  # Make prediction
    print(output[0])  # Print the prediction for debugging
    return jsonify(output[0])  # Return the prediction as a JSON response

# Route for standard form-based prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Collect the data from the form dynamically using the feature names
    data = [float(request.form[feature]) for feature in features]
    
    # Scale the input using the previously fitted scaler
    final_input = scalar.transform(np.array(data).reshape(1, -1))  # Scale the input
    print(final_input)  # For debugging: Print the transformed input
    
    # Make the prediction using the model
    output = regmodel.predict(final_input)[0]  # Get the predicted house price
    return render_template("home.html", features=features, prediction_text=f"The House price prediction is ${output:,.2f}")  # Display result on the page

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app with debugging enabled
