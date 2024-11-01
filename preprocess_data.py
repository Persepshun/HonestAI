import pandas as pd  # Data handling library
import re  # Regular expressions for text processing

def clean_text(text):
    """
    Cleans the text by removing URLs and special characters.
    This prepares text for analysis by simplifying it to lowercase letters.
    """
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabet characters
    return text.lower().strip()  # Standardize case and strip whitespace

def preprocess_gdelt_data(data):
    """
    Prepares GDELT data by cleaning content and assigning a target based on word count.
    Assumes GDELT data contains a 'title' column.
    """
    data["cleaned_content"] = data["title"].apply(clean_text)  # Clean content
    data["target"] = data["cleaned_content"].apply(lambda x: 1 if "technology" in x else 0)  # Assign binary target
    return data[["cleaned_content", "target"]]

def preprocess_common_crawl_data(data):
    """
    Processes Common Crawl data by cleaning content and assigning a synthetic target.
    Assumes Common Crawl data contains 'text' column with raw content.
    """
    data["cleaned_content"] = data["text"].apply(clean_text)
    data["target"] = data["cleaned_content"].apply(lambda x: 1 if "technology" in x else 0)
    return data[["cleaned_content", "target"]]
