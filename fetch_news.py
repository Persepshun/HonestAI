import requests  # For HTTP requests to fetch data
import pandas as pd  # Data manipulation library
from config import GDELT_BASE_URL  # URL for GDELT API
from bs4 import BeautifulSoup  # Parses HTML content

def fetch_gdelt_data(query="technology", timespan="1d", max_records=10):
    """
    Fetches news articles from the GDELT API based on a specified query and timespan.
    The GDELT API provides media analysis based on topics, events, and regions.
    """
    params = {
        "query": query,
        "mode": "artlist",
        "timespan": timespan,
        "maxrecords": max_records,
        "format": "json"
    }
    response = requests.get(GDELT_BASE_URL, params=params)
    data = response.json().get("articles", [])  # Extract articles from the JSON response
    return pd.DataFrame(data)  # Convert to DataFrame for analysis

def fetch_common_crawl_news(query="technology"):
    """
    Placeholder function for fetching news data from Common Crawl.
    Common Crawl data usually requires parsing of S3 data; this example retrieves HTML based on search term.
    """
    # Replace with actual Common Crawl URL for specific dataset
    url = f"https://example-commoncrawl-url.com/search?q={query}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data from Common Crawl with status code {response.status_code}")
        return pd.DataFrame()

    # Parse HTML and find relevant text
    soup = BeautifulSoup(response.content, "html.parser")
    texts = [element.text for element in soup.find_all("p") if query.lower() in element.text.lower()]
    return pd.DataFrame({"text": texts, "source": ["Common Crawl"] * len(texts)})
