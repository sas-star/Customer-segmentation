from flask import Flask, render_template, send_file
import pandas as pd
import pickle
from visuals import generate_visuals
import io

app = Flask(__name__)

@app.route('/')
def home():
    import os
    df = pd.read_csv('data/Cleaned_mall_customer_data.csv')

    with open("models/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    df['Cluster'] = kmeans.predict(X)

    # Performance optimization: only generate visuals if any are missing
    required_visuals = [
        "static/cluster_distribution.png",
        "static/3d_clusters.html",
        "static/income_spending.html",
        "static/age_spending.html",
        "static/gender_distribution.html",
        "static/box_plots.html",
        "static/radar_chart.html"
    ]
    if not all(os.path.exists(path) for path in required_visuals):
        generate_visuals(df)

    total_customers = len(df)
    avg_age = round(df['Age'].mean(), 2)
    avg_income = round(df['Annual Income (k$)'].mean(), 2)
    avg_spending = round(df['Spending Score (1-100)'].mean(), 2)

    profiles = df.groupby('Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean().round(2).to_dict('index')
    distances = kmeans.transform(X).min(axis=1)
    df['DistanceScore'] = distances.round(2)

    return render_template('index.html',
                           total_customers=total_customers,
                           avg_age=avg_age,
                           avg_income=avg_income,
                           avg_spending=avg_spending,
                           profiles=profiles,
                           customers=df.head(20).to_dict('records'))

@app.route('/download_profiles')
def download_profiles():
    df = pd.read_csv('data/Cleaned_mall_customer_data.csv')
    with open("models/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    df['Cluster'] = kmeans.predict(X)
    df['DistanceScore'] = kmeans.transform(X).min(axis=1).round(2)

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(io.BytesIO(output.getvalue().encode()),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="customer_profiles.csv")
@app.route('/test')
def test():
    return "<h1>Flask is working</h1>"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
