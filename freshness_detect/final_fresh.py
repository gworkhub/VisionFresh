import os
import pandas as pd
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from collections import defaultdict
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# Initialize Custom Vision Client
ENDPOINT = os.getenv("VISION_PREDICTION_ENDPOINT") 
prediction_key = os.getenv("VISION_PREDICTION_KEY")
project_id = "f15567d3-0803-4fb6-a205-d2b6fec9a29f"
publish_iteration_name = "fruitnveg"

# Authenticate the prediction client
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

# Dictionary to store images of fruits
product_images = {
    'Product 1': ['Freshness_detect/imgs/a1.jpg',
                  'Freshness_detect/imgs/a2.jpg',
                  'Freshness_detect/imgs/a3.jpg'], 
    'Product 2': ['Freshness_detect/imgs/b1.jpg',
                  'Freshness_detect/imgs/b2.jpg',
                  'Freshness_detect/imgs/b3.jpg'],
    'Product 3': ['freshness_detect/imgs/c1.jpg',
                  'Freshness_detect/imgs/c2.jpg',
                  'Freshness_detect/imgs/c3.jpg'],
}

# Function to analyze a single image
def analyze_image(image_path):
    print(f"  Analyzing image: {image_path}")
    with open(image_path, "rb") as image_contents:
        # Make the prediction
        results_api = predictor.classify_image(project_id, publish_iteration_name, image_contents.read())
    return results_api

# Store results
results = {}

# Process each product's images
for product, images in product_images.items():
    print(f"Processing {product}...")
    predictions_dict = defaultdict(list)

    with ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(analyze_image, image_path): image_path for image_path in images}

        for future in as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                results_api = future.result()
                # Store probabilities for each tag
                for prediction in results_api.predictions:
                    if prediction.probability > 0.10:  # Only consider predictions greater than 10%
                        predictions_dict[prediction.tag_name].append(prediction.probability * 100)
            except Exception as exc:
                print(f"{image_path} generated an exception: {exc}")

    # Aggregate results for the current product
    final_predictions = {}
    for tag, probabilities in predictions_dict.items():
        average_probability = sum(probabilities) / len(probabilities)
        final_predictions[tag] = average_probability

    # Store the final decision for this product
    results[product] = final_predictions
    print(f"  Completed processing for {product}.")

# Create a DataFrame to store the results
results_df = pd.DataFrame.from_dict(results, orient='index').fillna(0)

# Save results to an Excel file
output_file = 'freshness_predictions.xlsx'
results_df.to_excel(output_file, engine='openpyxl')

print(f"Results saved to {output_file}.")
