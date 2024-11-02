import csv
import requests

# Path to your existing keywords.csv file
CSV_FILE_PATH = "keywords.csv"

# Predefined list of categories to assign
CATEGORIES = [
    "education", "fish", "technology", "sports", "health", "finance", "politics", 
    "entertainment", "environment", "space", "food"
]

# Function to fetch a word from an online API (optional)
def fetch_word():
    try:
        response = requests.get("https://random-word-api.herokuapp.com/word?number=1")
        if response.status_code == 200:
            return response.json()[0]
    except requests.RequestException:
        print("Failed to fetch keyword. Using a fallback word.")
    return "defaultword"

# Function to add keywords to keywords.csv without random selection
def add_keywords(num_entries=10):
    new_entries = []
    for i in range(num_entries):
        category = CATEGORIES[i % len(CATEGORIES)]  # Cyclically select a category
        word = fetch_word()  # Fetch a word for the keyword
        new_entries.append((category, word))
    
    # Write new entries to the CSV file
    with open(CSV_FILE_PATH, mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_entries)

    print(f"Added {num_entries} new keywords to {CSV_FILE_PATH}.")

# Run the function to add 10 keywords (customize number as needed)
add_keywords(num_entries=10)
