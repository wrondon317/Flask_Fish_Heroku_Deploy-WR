import pickle
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model.
with open('fish_classifier.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    """
    For rendering webpage.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on webpage.
    """
    int_features = [x for x in request.form.values()]

    # Feature scaling.
    scaler = MinMaxScaler()
    input_scaled = scaler.fit_transform(np.array(int_features).reshape(-1, 1))

    prediction = model.predict(input_scaled.reshape(1,-1))

    if prediction:
        result = prediction[0]
        return render_template('index.html', prediction_text=f'The fish belong to species: {result}')
    else:
        return render_template('index.html', prediction_text='Model returned no predictions.')
    

if __name__ == '__main__':
    app.run()
