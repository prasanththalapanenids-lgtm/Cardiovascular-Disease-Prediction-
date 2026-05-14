from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        age_days = age * 365
        features = np.array([[age_days, gender, height, weight, ap_hi, ap_lo,
                              cholesterol, gluc, smoke, alco, active]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        result = 'Cardiovascular Disease Detected' if prediction == 1 else 'No Cardiovascular Disease'

        return render_template('result.html', result=result, probability=round(prob*100, 2))

    return render_template('predict.html')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
