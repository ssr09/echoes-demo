import os
import csv
import json
import random
from openai import OpenAI
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI()  # This will now use OPENAI_API_KEY from .env file

def chunk_text(file_path, chunk_size=3000, overlap=500):
    """Read and yield text chunks from file to avoid loading entire file in memory."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read(chunk_size + overlap)  # Initial read
        
        while text:
            # Find a good break point
            if len(text) >= chunk_size:
                end = min(chunk_size, len(text))
                
                # Look for last sentence break
                last_break = max(
                    text.rfind('. ', 0, end),
                    text.rfind('? ', 0, end),
                    text.rfind('! ', 0, end)
                )
                
                if last_break != -1:
                    end = last_break + 2  # Include the period and space
                
                chunk = text[:end]
                # Keep overlap for next chunk
                text = text[max(0, end - overlap):] + file.read(chunk_size)
            else:
                chunk = text
                text = ""
                
            yield chunk

def extract_quotes_with_ai(text):
    """Extract meaningful quotes using AI assistance."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts meaningful philosophical quotes from text. Identify 5-10 insightful quotes that contain wisdom or philosophical insight. Return your response as a JSON object with a single field 'quotes' containing an array of quote strings."},
            {"role": "user", "content": f"Extract meaningful philosophical quotes from this text:\n\n{text}"}
        ]
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get("quotes", [])

def get_quote_data(quote):
    """Get explanation and tags for a quote using OpenAI's API."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides explanations and tags for philosophical quotes. Return your response as a JSON object with fields 'explanation' and 'tags'."},
            {"role": "user", "content": f"Please provide an explanation (1-2 sentences) and 3-5 relevant tags for this quote: \"{quote}\""}
        ]
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get("explanation", ""), result.get("tags", [])

def get_embedding(text):
    """Get embedding for a text using OpenAI's API."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def process_quote(quote, author, csv_writer, json_list):
    """Process a single quote and save it directly to avoid memory buildup."""
    print(f"Processing quote: {quote[:40]}...")
    
    # Get explanation and tags
    explanation, tags = get_quote_data(quote)
    
    # Get embedding
    embedding = get_embedding(quote)
    
    # Generate random values for upvotes and popularity
    upvotes = random.randint(1, 20)
    popularity = round(random.uniform(0, 0.2), 2)
    
    # Create quote object
    quote_obj = {
        "quote": quote,
        "author": author,
        "explanation": explanation,
        "tags": tags,
        "upvotes": upvotes,
        "popularity": popularity,
        "embedding": embedding
    }
    
    # Add to JSON list
    json_list.append(quote_obj)
    
    # Write to CSV immediately
    csv_writer.writerow([
        quote,
        author,
        explanation,
        json.dumps(tags),
        upvotes,
        popularity,
        json.dumps(embedding)
    ])
    
    print(f"Processed quote: {quote[:40]}...")

def main():
    input_folder = "demo2"
    output_csv = "quotes.csv"
    output_json = "quotes.json"
    chunk_size = 3000
    overlap = 500
    
    # Check if folder exists
    if not os.path.exists(input_folder):
        print(f"Folder '{input_folder}' does not exist.")
        return
    
    # Create CSV file and writer
    csv_file = open(output_csv, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["quote", "author", "explanation", "tags", "upvotes", "popularity", "embedding"])
    
    # Initialize JSON list
    quotes_data = []
    
    # Process all text files in the folder
    file_count = 0
    chunk_count = 0
    quote_count = 0
    
    try:
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                file_count += 1
                file_path = os.path.join(input_folder, filename)
                
                # Extract author name from filename (remove extension)
                author = os.path.splitext(filename)[0]
                
                print(f"Processing file {file_count}: {filename}")
                
                # Process file in chunks
                for chunk in chunk_text(file_path, chunk_size, overlap):
                    chunk_count += 1
                    print(f"Processing chunk {chunk_count}")
                    
                    # Extract quotes from this chunk
                    quotes = extract_quotes_with_ai(chunk)
                    
                    # Process each quote immediately
                    for quote in quotes:
                        process_quote(quote, author, csv_writer, quotes_data)
                        quote_count += 1
        
        print(f"Processed {file_count} files, {chunk_count} chunks and found {quote_count} quotes")
        
        # Save JSON data
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(quotes_data, json_file, indent=2)
            
        print(f"Results saved to {output_csv} and {output_json}")
    
    finally:
        # Always close the CSV file
        csv_file.close()

if __name__ == "__main__":
    main() 