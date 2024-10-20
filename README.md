# VisionFresh

# OCR Analysis & Fruit and Vegetable Freshness

This repository contains two projects that leverage Azure's Cognitive Services for image analysis:

1. **Freshness Analysis**: Uses Azure's Custom Vision service to classify images of fruits and vegetables and assess their freshness.
2. **OCR and Data Extraction**: Utilizes Azure's Computer Vision service to extract text from product images, including prices and dates.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Data Output](#data-output)

## Features

- **Freshness Analysis**:

  - Classifies multiple images concurrently for efficient processing.
  - Outputs average probabilities for freshness predictions in an Excel file.
- **OCR and Data Extraction**:

  - Preprocesses images to enhance text recognition.
  - Extracts key information such as dates, prices, and quantities.
  - Normalizes quantities to a common unit (grams).
  - Saves results in separate Excel files for original and processed data.

## Requirements

- Python 3.6 or higher
- Azure Cognitive Services account with Custom Vision and Computer Vision
- Necessary Python packages:
  - `opencv-python`
  - `azure-cognitiveservices-vision-customvision`
  - `azure-cognitiveservices-vision-computervision`
  - `msrest`
  - `python-dotenv`
  - `pandas`
  - `openpyxl`

## Usage

1. For the **Freshness Analysis**:

   - Place your images in the specified directories in the script.
   - Run the script to get freshness predictions, which will be saved in an Excel file.
2. For the **OCR and Data Extraction**:

   - Place your product images in the `Ocr_extraction/imgs/` directory.
   - Run the script to extract text and data, which will be saved in two Excel files:
     - `original_product_data.xlsx`: Contains raw extracted data.
     - `processed_product_data.xlsx`: Contains structured data including MRP, quantities, and dates.

## Data Output

- **Freshness Analysis**: Results are saved in an Excel file containing average probabilities for each product's freshness.
- **OCR and Data Extraction**:
  - The **original product data** file includes combined text from processed images, detected dates, and prices.
  - The **processed data** file summarizes MRP, quantities (in grams), and relevant manufacturing and expiry dates.
