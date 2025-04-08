import os
import csv
import json
import random
import uuid
import time
import argparse
from openai import OpenAI
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI()  # This will now use OPENAI_API_KEY from .env file

def chunk_text(file_path, chunk_size=3000, overlap=500):
    """Read and yield text chunks from file to avoid loading entire file in memory."""
    try:
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
    except UnicodeDecodeError:
        # Try with a different encoding if utf-8 fails
        with open(file_path, 'r', encoding='latin-1') as file:
            text = file.read(chunk_size + overlap)
            
            while text:
                # Same logic as above
                if len(text) >= chunk_size:
                    end = min(chunk_size, len(text))
                    last_break = max(
                        text.rfind('. ', 0, end),
                        text.rfind('? ', 0, end),
                        text.rfind('! ', 0, end)
                    )
                    
                    if last_break != -1:
                        end = last_break + 2
                    
                    chunk = text[:end]
                    text = text[max(0, end - overlap):] + file.read(chunk_size)
                else:
                    chunk = text
                    text = ""
                    
                yield chunk

def api_call_with_retry(func, max_retries=3, backoff_factor=2):
    """Generic function to make API calls with exponential backoff retry logic."""
    retries = 0
    while retries < max_retries:
        try:
            return func()
        except Exception as e:
            retries += 1
            if retries == max_retries:
                print(f"Failed after {max_retries} retries: {e}")
                raise
            wait_time = backoff_factor ** retries
            print(f"API call failed: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

def extract_quotes_with_ai(text):
    """Extract meaningful quotes using AI assistance."""
    def call_api():
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a literary curator. From the following passage, extract up to 5 short, high-quality quotes that reflect timeless insight, poetic language, or philosophical value. Each quote can be up to 3 lines long, with natural line breaks. Return your response as a JSON object with a single field 'quotes' containing an array of quote strings. Preserve any natural paragraph breaks within quotes by including '\\n' characters."},
                {"role": "user", "content": f"Extract meaningful high-quality philosophical quotes (up to 3 lines each, with natural line breaks preserved) from this text:\n\n{text}"}
            ]
        )
        return response
    
    response = api_call_with_retry(call_api)
    try:
        result = json.loads(response.choices[0].message.content)
        # Ensure newlines in quotes are preserved
        quotes = [quote.replace('\\n', '\n') for quote in result.get("quotes", [])]
        return quotes
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing API response: {e}")
        return []

def get_quote_data(quote):
    """Get explanation and tags for a quote using OpenAI's API."""
    def call_api():
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides explanations and tags for philosophical quotes. Return your response as a JSON object with fields 'explanation' and 'tags'."},
                {"role": "user", "content": f"Please provide an explanation (upto 4 sentences) and 3-5 relevant tags for this quote: \"{quote}\""}
            ]
        )
        return response
    
    response = api_call_with_retry(call_api)
    try:
        result = json.loads(response.choices[0].message.content)
        return result.get("explanation", ""), result.get("tags", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing API response: {e}")
        return "", []

def get_embedding(text):
    """Get embedding for a text using OpenAI's API."""
    def call_api():
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response
    
    response = api_call_with_retry(call_api)
    return response.data[0].embedding

def generate_unique_id(quote, author):
    """Generate a unique ID for a quote based on its content and author."""
    # Create a unique identifier using a hash of quote and author
    unique_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{quote}-{author}"))
    # Use only the first 8 characters to keep it short but still unique
    return f"q-{unique_hash[:8]}"

def is_valid_quote(quote, min_length=20):
    """Check if a quote is valid (not too short and has actual content)."""
    # Remove whitespace and check length
    cleaned_quote = quote.strip()
    if len(cleaned_quote) < min_length:
        return False
    
    # Check if it has actual words
    word_count = len(re.findall(r'\b\w+\b', cleaned_quote))
    if word_count < 4:  # At least 4 words
        return False
        
    return True

def evaluate_quote_quality(quote):
    """Evaluate the quality of a quote on a scale of 1-10."""
    def call_api():
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a literary critic with expertise in evaluating quotes. Rate the following quote on a scale of 1-10 based on its timelessness, literary value, philosophical depth, and cultural impact. Try to not give value to the where the quote is from in the larger context and evaluate the quote primarily based on its own merit. Return ONLY a JSON object with fields 'score' (number 1-10) and 'reasoning' (brief justification)."},
                {"role": "user", "content": f"Evaluate this quote on a scale of 1-10. Return ONLY the score and brief reasoning in JSON format: \"{quote}\""}
            ]
        )
        return response
    
    try:
        response = api_call_with_retry(call_api)
        result = json.loads(response.choices[0].message.content)
        score = result.get("score", 0)
        reasoning = result.get("reasoning", "")
        print(f"Quote quality score: {score}/10 - {reasoning[:50]}...")
        return score, reasoning
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error evaluating quote quality: {e}")
        return 0, "Error in evaluation"

def process_quote(quote, author, csv_writer, json_list, processed_hashes):
    """Process a single quote and save it directly to avoid memory buildup."""
    # Skip invalid quotes
    if not is_valid_quote(quote):
        print(f"Skipping invalid quote: {quote[:40]}...")
        return False
    
    # Skip duplicates by checking content hash
    quote_hash = hash(quote.strip().lower())
    if quote_hash in processed_hashes:
        print(f"Skipping duplicate quote: {quote[:40]}...")
        return False
    
    # Evaluate quote quality
    quality_score, quality_reasoning = evaluate_quote_quality(quote)
    if quality_score < 7:
        print(f"Skipping low-quality quote (score {quality_score}/10): {quote[:40]}...")
        return False
    
    processed_hashes.add(quote_hash)
    print(f"Processing quote: {quote[:40]}...")
    
    # Generate unique ID
    quote_id = generate_unique_id(quote, author)
    
    # Get explanation and tags
    explanation, tags = get_quote_data(quote)
    
    # Get embedding
    embedding = get_embedding(quote)
    
    # Generate random values for upvotes and popularity
    upvotes = random.randint(1, 20)
    popularity = round(random.uniform(0, 0.2), 2)
    
    # Create quote object for JSON (preserve newlines)
    quote_obj = {
        "id": quote_id,
        "quote": quote,
        "author": author,
        "explanation": explanation,
        "tags": tags,
        "quality_score": quality_score,
        "quality_reasoning": quality_reasoning,
        "upvotes": upvotes,
        "popularity": popularity,
        "embedding": embedding
    }
    
    # Add to JSON list
    json_list.append(quote_obj)
    
    # For CSV, replace literal newlines with \n representation for display purposes
    csv_safe_quote = quote.replace('\n', '\\n')
    
    # Write to CSV immediately
    csv_writer.writerow([
        quote_id,
        csv_safe_quote,
        author,
        explanation,
        json.dumps(tags),
        quality_score,
        quality_reasoning,
        upvotes,
        popularity,
        json.dumps(embedding)
    ])
    
    print(f"Processed quote: {quote[:40]}...")
    return True

def save_batch_json(quotes_data, output_json, batch_size=100):
    """Save JSON data in batches to prevent memory issues with large datasets."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    
    # Save JSON data with proper handling of newlines
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(quotes_data, json_file, ensure_ascii=False, indent=2)
    
    print(f"Batch of {len(quotes_data)} quotes saved to {output_json}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract quotes from text files.')
    parser.add_argument('--input', '-i', default='demo2', help='Input folder containing text files')
    parser.add_argument('--output-csv', default='quotes.csv', help='Output CSV file')
    parser.add_argument('--output-json', default='quotes.json', help='Output JSON file')
    parser.add_argument('--chunk-size', type=int, default=3000, help='Size of text chunks for processing')
    parser.add_argument('--overlap', type=int, default=500, help='Overlap between chunks')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for saving JSON')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    input_folder = args.input
    output_csv = args.output_csv
    output_json = args.output_json
    chunk_size = args.chunk_size
    overlap = args.overlap
    batch_size = args.batch_size
    
    # Check if folder exists
    if not os.path.exists(input_folder):
        print(f"Folder '{input_folder}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    
    # Create CSV file and writer with the right quoting settings
    csv_file = open(output_csv, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    csv_writer.writerow(["id", "quote", "author", "explanation", "tags", "quality_score", "quality_reasoning", "upvotes", "popularity", "embedding"])
    
    # Initialize JSON list and track processed quotes to avoid duplicates
    quotes_data = []
    processed_hashes = set()
    
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
                        if process_quote(quote, author, csv_writer, quotes_data, processed_hashes):
                            quote_count += 1
                            
                            # Save in batches to manage memory
                            if quote_count % batch_size == 0:
                                save_batch_json(quotes_data, output_json)
        
        print(f"Processed {file_count} files, {chunk_count} chunks and found {quote_count} quotes")
        
        # Save final JSON data
        save_batch_json(quotes_data, output_json)
            
        print(f"Results saved to {output_csv} and {output_json}")
    
    except KeyboardInterrupt:
        print("Process interrupted. Saving current results...")
        save_batch_json(quotes_data, output_json)
        print(f"Partial results saved to {output_csv} and {output_json}")
    
    finally:
        # Always close the CSV file
        if 'csv_file' in locals():
            csv_file.close()

if __name__ == "__main__":
    main() 