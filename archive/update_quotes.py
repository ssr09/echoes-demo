import json
import csv
import random

# Load existing JSON data
with open('quotes.json', 'r', encoding='utf-8') as file:
    quotes = json.load(file)

# Add author, upvotes, and popularity to each quote
for quote in quotes:
    # Add author attribution
    quote["author"] = "Marcus Aurelius"
    
    # Initialize upvotes with random values (between 0-100 for demo purposes)
    quote["upvotes"] = random.randint(0, 100)
    
    # Initialize popularity score (could be calculated based on upvotes and time)
    # For now, set it as a normalized value between 0-1
    quote["popularity"] = round(quote["upvotes"] / 100, 2)

# Save updated JSON
with open('quotes_updated.json', 'w', encoding='utf-8') as file:
    json.dump(quotes, file, indent=2)

# Save updated CSV
with open('quotes_updated.csv', 'w', newline='', encoding='utf-8') as file:
    # Determine all fields to include in the CSV
    fieldnames = ["quote", "author", "explanation", "tags", "upvotes", "popularity", "embedding"]
    
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    
    for quote in quotes:
        # Create a row dict with all fields
        row = {
            "quote": quote["quote"],
            "author": quote["author"],
            "explanation": quote["explanation"],
            "tags": json.dumps(quote["tags"]),
            "upvotes": quote["upvotes"],
            "popularity": quote["popularity"],
            "embedding": json.dumps(quote["embedding"])
        }
        writer.writerow(row)

print("Updated quotes saved to quotes_updated.json and quotes_updated.csv")
print(f"Added author, upvotes, and popularity data to {len(quotes)} quotes") 