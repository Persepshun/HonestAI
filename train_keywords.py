import csv
import random
import requests

# Path to your existing keywords.csv file
CSV_FILE_PATH = "keywords.csv"

# Predefined list of categories to assign
CATEGORIES = [
    "education", "fish", "technology", "sports", "health", "finance", "politics", 
    "entertainment", "environment", "space", "food"
]

# Function to get a random word from an online API (optional)
def fetch_random_word():
    try:
        response = requests.get("https://random-word-api.herokuapp.com/word?number=1")
        if response.status_code == 200:
            return response.json()[0]
    except requests.RequestException:
        print("Failed to fetch keyword. Using a fallback word.")
    return "defaultword"

# Function to add random keywords to keywords.csv
def add_random_keywords(num_entries=10):
    new_entries = []
    for _ in range(num_entries):
        category = random.choice(CATEGORIES)  # Randomly select a category
        word = fetch_random_word()  # Fetch a random word for the keyword
        new_entries.append((category, word))
    
    # Write new entries to the CSV file
    with open(CSV_FILE_PATH, mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_entries)

    print(f"Added {num_entries} new keywords from saved_models to {CSV_FILE_PATH}.")

# Run the function to add 10 random keywords (customize number as needed)
add_random_keywords(num_entries=10)
