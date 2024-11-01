from flask import Flask, render_template, request, redirect, url_for
import bias_detection
import ethical_screening
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Route for the landing page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the URL checker page
@app.route('/url_checker', methods=['GET', 'POST'])
def url_checker():
    if request.method == 'POST':
        url = request.form['url']
        return redirect(url_for('analyze_url', url=url))
    return render_template('url_checker.html')

# Route for analyzing the URL
@app.route('/analyze_url')
def analyze_url():
    url = request.args.get('url')
    
    # Fetch content from the URL
    response = requests.get(url)
    if response.status_code != 200:
        return f"Failed to retrieve content from {url}", 400
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    text_content = soup.get_text()  # Extract text content from the page
    
    # Placeholder DataFrame for content processing (can be replaced by real data)
    data = pd.DataFrame([{'content': text_content}])

    # Preprocess data and run bias detection and ethical checks (example using dummy data)
    X, y = bias_detection.preprocess_data(data)
    model = LogisticRegression()
    model.fit(X, y)

    # Run bias detection
    y_pred = model.predict(X)
    sensitive_feature = X[['race_ White', 'sex_ Male']]
    dp_difference, dp_difference_sex = bias_detection.evaluate_fairness(y, y_pred, sensitive_feature)
    disparate_impact, statistical_parity = bias_detection.evaluate_aif360_metrics(X, y)

    # Run ethical screening
    ethical_screening.ethical_checklist(X, model)

    # Collect results
    results = {
        'dp_difference_race': dp_difference,
        'dp_difference_sex': dp_difference_sex,
        'disparate_impact': disparate_impact,
        'statistical_parity': statistical_parity
    }
    
    # Render results in a new template
    return render_template('url_analysis_result.html', url=url, results=results)

if __name__ == '__main__':
    app.run(debug=True)
