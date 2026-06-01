from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load dataset for charts
df = pd.read_csv('cardio_train.csv', sep=';')
df['bmi'] = df['weight'] / ((df['height']/100)**2)

# Convert age from days to years for charts
df['age_years'] = (df['age'] / 365).astype(int)

# Generate Plotly figures and convert to HTML strings
def create_charts():
    charts = {}

    # 1. Age distribution
    fig_age = px.histogram(df, x='age_years', nbins=50, color='cardio',
                           title='Age Distribution vs Cardiovascular Disease')
    charts['age'] = pio.to_html(fig_age, full_html=False)

    # 2. Gender vs Cardio counts
    fig_gender = px.histogram(df, x='gender', color='cardio', barmode='group',
                              title='Gender vs Cardiovascular Disease')
    fig_gender.update_xaxes(tickvals=[0,1], ticktext=['Female','Male'])
    charts['gender'] = pio.to_html(fig_gender, full_html=False)

    # 3. Cholesterol levels
    fig_chol = px.histogram(df, x='cholesterol', color='cardio', barmode='group',
                            title='Cholesterol Levels vs Cardiovascular Disease')
    fig_chol.update_xaxes(tickvals=[1,2,3], ticktext=['Normal','Above Normal','Well Above Normal'])
    charts['cholesterol'] = pio.to_html(fig_chol, full_html=False)

    # 4. Glucose levels
    fig_gluc = px.histogram(df, x='gluc', color='cardio', barmode='group',
                            title='Glucose Levels vs Cardiovascular Disease')
    fig_gluc.update_xaxes(tickvals=[1,2,3], ticktext=['Normal','Above Normal','Well Above Normal'])
    charts['gluc'] = pio.to_html(fig_gluc, full_html=False)

    # 5. Blood pressure scatter
    fig_bp = px.scatter(df, x='ap_hi', y='ap_lo', color='cardio',
                        title='Systolic vs Diastolic Blood Pressure', opacity=0.6)
    charts['bp'] = pio.to_html(fig_bp, full_html=False)

    # 6. BMI distribution
    fig_bmi = px.histogram(df, x='bmi', nbins=50, color='cardio',
                           title='BMI Distribution vs Cardiovascular Disease')
    charts['bmi'] = pio.to_html(fig_bmi, full_html=False)

    # 7. Correlation heatmap
    corr = df[['age_years','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio']].corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis',
        zmin=-1, zmax=1))
    fig_corr.update_layout(title='Feature Correlation Heatmap')
    charts['corr'] = pio.to_html(fig_corr, full_html=False)

    return charts

charts = create_charts()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
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
        
        except Exception as e:
            return f"Error: {e}"

    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', charts=charts)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
