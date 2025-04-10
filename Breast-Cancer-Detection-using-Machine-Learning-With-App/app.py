from flask import Flask, render_template, request
import numpy as np
import pickle

# For the model in the notebook and dataset folder
model = pickle.load(open('notebook and dataset/model.pkl', 'rb'))

expected_features = model.n_features_in_
print(f"Model expects {expected_features} features.")

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', message=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        features = request.form['feature']
        features_lst = features.split(',')
        np_features = np.asarray(features_lst, dtype=np.float32)

        # Check if feature count matches model requirement
        if np_features.shape[0] != expected_features:
            error_message = [
                f"Error: Model expects {expected_features} features, but got {np_features.shape[0]}."
            ]
            return render_template('index.html', message=error_message)

        # Perform prediction
        pred = model.predict(np_features.reshape(1, -1))
        output = "Cancerous" if pred[0] == 1 else "Not Cancerous"
        
        return render_template('index.html', message=[output])

    except Exception as e:
        error_message = [f"Error: {str(e)}"]
        return render_template('index.html', message=error_message)


if __name__ == '__main__':
    app.run(debug=True)