import cv2
import os
import time
import re
import string
from datetime import datetime
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import pandas as pd  # Import pandas

# Load environment variables from .env file
load_dotenv()

# Azure Cognitive Services credentials
subscription_key = os.getenv('VISION_KEY')  # Load from .env
endpoint = os.getenv('VISION_ENDPOINT')  # Load from .env

# Initialize Azure Computer Vision client
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

def preprocess_image(input_image_path):
    start_time = time.time()
    image = cv2.imread(input_image_path)

    if image is None:
        raise FileNotFoundError(f"Error: Unable to load image from {input_image_path}. Please check the file path.")

    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10
    )

    # Save the preprocessed image
    preprocessed_image_path = input_image_path.replace('.jpg', '_preprocessed.jpg')
    cv2.imwrite(preprocessed_image_path, adaptive_thresh)
    print(f"Preprocessed image saved at {preprocessed_image_path}")

    end_time = time.time()
    print(f"Preprocessing time: {end_time - start_time:.2f} seconds")

    return preprocessed_image_path

def read_text_from_image(image_path):
    start_time = time.time()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if os.path.getsize(image_path) > 4 * 1024 * 1024:  # 4 MB limit
        print(f"Image too large: {image_path}. Skipping...")
        return ""

    result = None  # Initialize result variable

    for attempt in range(5):
        try:
            with open(image_path, 'rb') as image_stream:
                result = client.read_in_stream(image_stream, raw=True)
            break  # Break if successful
        except Exception as e:
            if "Too Many Requests" in str(e):
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
            elif "Bad Request" in str(e):
                print(f"Bad Request for image: {image_path}. Skipping...")
                return ""
            else:
                raise e  # Re-raise other exceptions

    if result is None:
        print(f"Failed to process image: {image_path}. No valid result.")
        return ""  # Return empty string if no result was obtained

    operation_location = result.headers['Operation-Location']
    operation_id = operation_location.split('/')[-1]

    print(f"Waiting for text recognition results for {image_path}...")
    while True:
        read_result = client.get_read_result(operation_id)
        if read_result.status.lower() != 'running':
            break

    extracted_text = []
    if read_result.status == 'succeeded':
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                extracted_text.append(line.text)
                print(line.text)
    else:
        print(f"No text recognized in {image_path}.")

    return ' '.join(extracted_text)

def process_single_image(image_path):
    # Preprocess the image and read text
    preprocessed_image_path = preprocess_image(image_path)
    original_text = read_text_from_image(image_path)
    preprocessed_text = read_text_from_image(preprocessed_image_path)

    # Combine both texts
    return original_text + ' ' + preprocessed_text

def process_product_images(product_images):
    combined_text = ''
    
    # Using ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(process_single_image, product_images))

    # Combine all results
    combined_text = ' '.join(results)
    
    return combined_text.strip()

def clean_ocr_text(ocr_text):
    cleaned_text = ''.join(filter(lambda x: x in string.printable, ocr_text))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text = re.sub(r'\s+([.,!?;])', r'\1', cleaned_text)
    cleaned_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', cleaned_text)
    cleaned_text = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', cleaned_text)
    cleaned_text = re.sub(r'(?<=[\d])(?=[a-zA-Z])', ' ', cleaned_text)
    return cleaned_text

def detect_mrp(text):
    patterns = [
        r"(?:[Mm][Rr][Pp]\D*)(\d+)",  
        r"(?:[Pp][Rr][Ii][Cc][Ee]\D*)(\d+)",  
        r"(\d+)\s*[Rr][Ss]",  
        r"(\d+)\s*[Rr][Uu][Pp][Ee][Ee]",  
        r"(\d+)\s*[Ii][Nn][Rr]",  
    ]
    
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))
    
    return [int(match) for match in matches]

# Extended regex for various date formats
regex = r"""
    # Match dates like 26/8/24 or 26/08/2024 (day/month/year)
    (\b\d{1,2}([\/\-.])\d{1,2}\2\d{2,4}\b) |
    # Match dates like 10.09.24 (day.month.year with two-digit year)
    (\b\d{1,2}([.])\d{1,2}\2\d{2}\b) |  
    # Match month/year combinations like 10/2025
    (\b\d{1,2}([\/\-.])\d{4}\b) |
    # Match full dates with month names, including ordinals
    (\b\d{1,2}(?:st|nd|rd|th)?\s*(January|February|March|April|May|June|July|August|
    September|October|November|December)\s?\d{2,4}\b) |
    # Match full dates with month names without ordinals
    (\b\d{1,2}\s+(January|February|March|April|May|June|July|August|
    September|October|November|December)\s?\d{2,4}\b) |
    # Match compact dates like 18APR2024
    (\b\d{1,2}(?:[A-Za-z]{3})\d{4}\b) |
    # Match dates with three-letter month abbreviations and spaces
    (\b\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\b) |
    # Match month-year combinations like May/27
    (\b(January|February|March|April|May|June|July|August|
    September|October|November|December)[\/\-.]?\s?\d{2,4}\b) |
    # Match ISO-like formats such as 2024-09-23
    (\b\d{4}([\/\-.])\d{1,2}\2\d{1,2}\b) |
    # Match just month names and two- or four-digit years
    (\b(January|February|March|April|May|June|July|August|
    September|October|November|December)[\/\-.]?\d{2,4}\b) |
    # Match ordinal dates like 23rd of October 2024
    (\b\d{1,2}(?:st|nd|rd|th)?\s+of\s+(January|February|March|April|May|June|July|August|
    September|October|November|December)\s?\d{2,4}\b) |
    # Match abbreviated month/year combinations like JUL/26
    (\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\/]\d{2}\b)
"""

date_regex = re.compile(regex, re.VERBOSE | re.IGNORECASE)

def parse_date(date_str):
    try:
        # Handle formats like JUL/26 or AUG/24
        if re.match(r'^[A-Z]{3}\/\d{2}$', date_str):
            month_abbrev, year = date_str.split('/')
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            month = month_map[month_abbrev.upper()]
            year = int(f"20{year}")  # Assume 2000s for 2-digit years
            if year < 2000:
                return None  # Skip years before 2000
            return datetime(year=year, month=month, day=1)

        # Handle month/year formats like May/26 or May 2024
        elif re.match(r'\d{1,2}[\/\-.]\d{4}', date_str):
            month, year = re.split(r'[\/\-\.]', date_str)
            year = int(year)
            if year < 2000:
                return None  # Skip years before 2000
            return datetime(year=year, month=int(month), day=1)

        # Handle month and two-digit year formats like May/27 or May2024
        elif re.match(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)[\/\-.]?\d{2}$', date_str):
            month_str, year = re.split(r'[\/\-.]', date_str)
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            year = int(f"20{year}")  # Assume 2000s for 2-digit years
            if year < 2000:
                return None  # Skip years before 2000
            return datetime(year=year, month=month_map[month_str], day=1)

        # Handle dd/mm/yy or dd/mm/yyyy formats
        elif '/' in date_str or '-' in date_str or '.' in date_str:
            parts = re.split(r'[\/\-.]', date_str)
            if len(parts) == 3:
                day, month, year = parts
                if len(year) == 2:  # Assume 21st century for two-digit year
                    year = f"20{year}"
                year = int(year)
                if year < 2000:
                    return None  # Skip dates before 2000
                return datetime(year=year, month=int(month), day=int(day))

    except ValueError:
        return None

    return None

# Utility function to convert quantities to a common unit (grams)
import re

# Utility function to convert quantities to a common unit (grams)
def normalize_quantity(quantity_str):
    """
    Normalize quantities to grams.
    E.g., 100mg -> 0.1g, 1kg -> 1000g, 50g -> 50g
    """
    # Define conversion factors (everything to grams)
    conversions = {
        'g': 1,
        'kg': 1000,
        'mg': 0.001,
        'ml': 1,  # Assuming density of 1 for liquids (like water)
        'cl': 10,
        'dl': 100,
        'pieces': 1,  # Treat as 1 piece = 1 gram for normalization
        'unit': 1,
        'count': 1
    }

    # Regular expression to capture quantity and unit
    match = re.search(r'(\d+)\s*(g|kg|mg|ml|cl|dl|pieces?|unit|count)', quantity_str, re.IGNORECASE)
    if match:
        value = int(match.group(1))  # Extract numeric value
        unit = match.group(2).lower()  # Extract unit

        # Convert value to grams based on unit
        normalized_value = value * conversions.get(unit, 1)  # Default to 1 if unit not found
        
        return normalized_value  # Return the normalized value in grams

    return 0 



def main():
    # Define the products and their respective images
    products = {
        'Product1': ['Ocr_extraction/imgs/1.jpg', 
                     'Ocr_extraction/imgs/2.jpg',
                     'Ocr_extraction/imgs/3.jpg'],
        'Product2': ['Ocr_extraction/imgs/4.jpg',
                     'Ocr_extraction/imgs/5.jpg',
                     'Ocr_extraction/imgs/6.jpg'],
        'Product3': ['Ocr_extraction/imgs/7.jpg',
                     'Ocr_extraction/imgs/8.jpg'],
    }

    all_product_data = []
    processed_data = []

    for product, images in products.items():
        print(f"Processing {product}...")
        combined_text = process_product_images(images)
        
        # Clean the combined text
        cleaned_text = clean_ocr_text(combined_text)
        
        # Find all matches for dates
        matches = re.findall(date_regex, cleaned_text)
        found_dates = [match for match_group in matches for match in match_group if match]
        
        # Parse and filter valid dates
        parsed_dates = [parse_date(date) for date in found_dates]
        parsed_dates = [date for date in parsed_dates if date is not None]
        
        # Sorting dates
        sorted_dates = sorted(parsed_dates)
        
        # Formatting sorted dates back to string for output
        sorted_dates_str = [date.strftime('%d %B %Y') for date in sorted_dates]
        
        # Detect MRP
        detected_mrp = detect_mrp(cleaned_text)
        
        # Regex pattern for detecting quantities
        pattern = r'\b(?:\d{1,5}\s*\/\s*\d{1,5}|\d{1,5}\s*\d*\s*\/\s*\d{1,5}|\d{1,5})\s*(?:g|kg|mg|l|ml|cl|dl|pieces?|packs?|units?|counts?|ct|ea|each|set)\b'
        detected_quantities = re.findall(pattern, cleaned_text, re.IGNORECASE)
        
        # Normalize quantities to grams
        normalized_quantities = [normalize_quantity(q) for q in detected_quantities]
        
        # Store the data
        all_product_data.append({
            'Product': product,
            'Dates': ', '.join(sorted_dates_str),
            'MRP': ', '.join(map(str, detected_mrp)),
            'Quantities': ', '.join(detected_quantities),  # Keep original for reference if needed
        })

        # Extract relevant data for new Excel
        if detected_mrp:
            # Use the normalized quantities in the processed data
            processed_data.append({
                'Product': product,
                'MRP (in ₹)': detected_mrp[0],
                'Quantity (in g)': max(normalized_quantities),  # Use normalized quantities
                'Manufacturing Date': min(parsed_dates).strftime('%d %B %Y') if len(parsed_dates) > 0 else None,
                'Expiry Date': max(parsed_dates).strftime('%d %B %Y') if len(parsed_dates) > 0 else None,
            })


    # Create a DataFrame and write to Excel (original data)
    df_original = pd.DataFrame(all_product_data)
    df_original.to_excel('original_product_data.xlsx', index=False)
    
    # Create a DataFrame for the processed data and write to another Excel file
    df_processed = pd.DataFrame(processed_data)
    df_processed.to_excel('processed_product_data.xlsx', index=False)
    
    print("Data stored in both original_product_data.xlsx and processed_product_data.xlsx")

if __name__ == "__main__":
    main()
