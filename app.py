from flask import Flask, render_template, request, redirect, url_for
import bias_detection  # Custom module for bias detection functions
import ethical_screening  # Custom module for ethical AI checks
import requests  # For sending HTTP requests to external URLs
from bs4 import BeautifulSoup  # For parsing HTML content to extract text
import pandas as pd  # Data analysis library for handling dataframes
from sklearn.linear_model import LogisticRegression  # Machine learning model for analysis
from fetch_news import fetch_gdelt_data, fetch_common_crawl_news  # Custom functions to fetch data from GDELT and Common Crawl
from preprocess_data import preprocess_gdelt_data, preprocess_common_crawl_data  # Custom preprocessing functions for GDELT and Common Crawl data

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppresses warnings for cleaner output

# Initialize Flask app
app = Flask(__name__)

# Load keywords from CSV for keyword-based classification
keywords_df = pd.read_csv("keywords.csv")  # Dataframe to store categories and keywords

def classify_text_by_keywords(text):
    """
    Classifies text into categories based on keyword matches.
    The function scans the text for predefined keywords in 'keywords.csv'
    and categorizes the text based on the category with the most matches.
    """
    text = text.lower()  # Ensure case consistency
    category_counts = {category: 0 for category in keywords_df['category'].unique()}  # Initialize counter for each category

    # Check for keywords in each category and count occurrences
    for _, row in keywords_df.iterrows():
        if row['keyword'] in text:
            category_counts[row['category']] += 1

    # Choose the category with the highest keyword count or return 'other' if no keywords match
    primary_category = max(category_counts, key=category_counts.get)
    return primary_category if category_counts[primary_category] > 0 else "other"

# Route for landing page
@app.route('/')
def index():
    """Displays the homepage of the Ethical AI Toolkit"""
    return render_template('index.html')

# Route for URL checker page
@app.route('/url_checker', methods=['GET', 'POST'])
def url_checker():
    """Handles form submission for checking a URL's content and analyzing it."""
    if request.method == 'POST':
        url = request.form['url']  # Get the URL from form data
        return redirect(url_for('analyze_url', url=url))  # Redirect to analysis route
    return render_template('url_checker.html')

# Route for analyzing content from a single URL
@app.route('/analyze_url')
def analyze_url():
    """
    Fetches content from a given URL, classifies it, preprocesses it, and applies bias detection and ethical screening.
    """
    url = request.args.get('url')
    
    # Request and check response status
    response = requests.get(url)
    if response.status_code != 200:
        return f"Failed to retrieve content from {url}", 400
    
    # Extract text from HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    text_content = soup.get_text()

    # Classify content based on keywords and set up data for analysis
    category = classify_text_by_keywords(text_content)
    data = pd.DataFrame([{'content': text_content, 'category': category}])

    # Generate a synthetic target for model training based on the category
    data['target'] = data['category'].apply(lambda x: 1 if x == "politics" else 0)  # Example: 'politics' category is class 1

    # Preprocess and handle for model training and bias analysis
    X, y = bias_detection.preprocess_data(data)

    # Return message if not enough classes are present for training
    if len(y.unique()) < 2:
        results = {
            'message': "Insufficient data classes for meaningful analysis. The target variable needs at least two unique classes."
        }
        return render_template('url_analysis_result.html', url=url, results=results)

    # Train the model on available data
    model = LogisticRegression()
    model.fit(X, y)

    # Generate predictions and perform fairness and ethical analysis
    y_pred = model.predict(X)
    sensitive_feature = X[['char_count', 'word_count']]
    dp_difference, dp_difference_sex = bias_detection.evaluate_fairness(y, y_pred, sensitive_feature)
    disparate_impact, statistical_parity = bias_detection.evaluate_aif360_metrics(X, y)
    ethical_screening.ethical_checklist(X, model)

    # Package results for display
    results = {
        'dp_difference_race': dp_difference,
        'dp_difference_sex': dp_difference_sex,
        'disparate_impact': disparate_impact,
        'statistical_parity': statistical_parity,
        'category': category
    }
    
    return render_template('url_analysis_result.html', url=url, results=results)

# Route for handling GDELT and Common Crawl searches
@app.route('/search', methods=['GET', 'POST'])
def search():
    """
    Handles the submission and analysis of data from GDELT and Common Crawl sources
    based on a user-provided search query.
    """
    if request.method == 'POST':
        query = request.form['query']  # Retrieve search query
        source = request.form['source']  # Determine data source: GDELT or Common Crawl

        # Fetch and preprocess data based on the selected source
        if source == 'gdelt':
            raw_data = fetch_gdelt_data(query=query)
            processed_data = preprocess_gdelt_data(raw_data)
        elif source == 'common_crawl':
            raw_data = fetch_common_crawl_news(query=query)
            processed_data = preprocess_common_crawl_data(raw_data)
        else:
            return "Invalid source selected.", 400

        # Check for enough unique classes in the target variable
        if len(processed_data["target"].unique()) < 2:
            results = {
                'message': "Insufficient data classes for meaningful analysis. The data must contain at least two unique classes."
            }
            return render_template('search_results.html', results=results)

        # Train model on processed data
        X = processed_data["cleaned_content"].apply(len).values.reshape(-1, 1)
        y = processed_data["target"]
        model = LogisticRegression().fit(X, y)

        # Analyze model for fairness and perform ethical checks
        y_pred = model.predict(X)
        dp_difference, dp_difference_sex = bias_detection.evaluate_fairness(y, y_pred, X)
        disparate_impact, statistical_parity = bias_detection.evaluate_aif360_metrics(X, y)
        ethical_screening.ethical_checklist(X, model)

        # Store results for display
        results = {
            'dp_difference_race': dp_difference,
            'dp_difference_sex': dp_difference_sex,
            'disparate_impact': disparate_impact,
            'statistical_parity': statistical_parity
        }
        return render_template('search_results.html', results=results)

    return render_template('search.html')

# Run the application
if __name__ == '__main__':
    app.run(debug=True)  # Enables debug mode for development
