import os
import json
import requests
import pandas as pd
from typing import List

def download_nhtsa(years: List[int], make: str = "FORD", file_name: str = 'full_data') -> None:
    """
    Fetch and save vehicle complaint data from the NHTSA (National Highway Traffic Safety Administration) API 
    for multiple years.

    Parameters:
    years List[int]): A list of model years to fetch (e.g., [2020, 2021, 2022]).
    
    make (str): The manufacturer (brand) of the vehicle (default is "FORD").
    """
    
    # Base URL to fetch vehicle models and complaints
    url_complaints = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
    
    # List to store all data
    all_data = []

    try:
        for year in years:
            print(f"Fetching models for year {year} and make {make}...")

            # URL to fetch models
            url_models = f"https://api.nhtsa.gov/products/vehicle/models?modelYear={year}&make={make}&issueType=c"

            # Make the request to get the models
            response = requests.get(url_models)
            response.raise_for_status()  # Raise exceptions for HTTP errors
            models_data = response.json()
            
            if "results" in models_data:
                models = models_data["results"]
                print(f"{len(models)} models found for the year {year} and make {make}.")
                
                for model_info in models:
                    model = model_info.get("model")
                    if model:
                        print(f"Fetching data for model: {model} (Year: {year})")
                        
                        # Fetch complaints data
                        response_complaints = requests.get(
                            f"{url_complaints}?make={make}&model={model}&modelYear={year}"
                        )
                        response_complaints.raise_for_status()
                        complaints_data = response_complaints.json()
                        
                        # Add data to the full dataset
                        all_data.append({
                            "year": year,
                            "make": make,
                            "model": model,
                            "complaints": complaints_data.get("results", [])
                        })
            else:
                print(f"No models found for the year {year} and make {make}.")

        if not all_data:
            print("No data was collected. Exiting function.")
            return

        # Save all data to a JSON file
        json_filename = f"data/raw/{file_name}.json"
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(all_data, json_file, ensure_ascii=False, indent=4)
        print(f"All data saved to: {json_filename}")

        # Convert data to DataFrame
        records = []
        for entry in all_data:
            year = entry["year"]
            make = entry["make"]
            model = entry["model"]
            for complaint in entry["complaints"]:
                complaint_data = {
                    "Year": year,
                    "Make": make,
                    "Model": model,
                    **complaint,  # Add all complaint fields
                }
                records.append(complaint_data)
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(records)
        csv_filename = f"data/raw/{file_name}.csv"
        df.to_csv(csv_filename, index=False, encoding="utf-8")
        print(f"All data saved to: {csv_filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error making the request: {e}")

if __name__ == '__main__':
    file_name = 'full_data_2020_2025_FORD'
    path = f'data/raw/{file_name}.csv'

if not os.path.isfile(path):
    download_nhtsa(years= list(range(2020, 2026)), file_name=file_name)
