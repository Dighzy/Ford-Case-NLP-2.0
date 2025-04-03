import os
import sys
import requests
import subprocess
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

def download_nhtsa(cfg: DictConfig) -> None:
    """
    - Fetch and save vehicle complaint data from the NHTSA (National Highway Traffic Safety Administration) API
    for multiple years.

    ### Parameters:
    - **years List[int])**: A list of model years to fetch (e.g., [2020, 2021, 2022]).
    
    - **make (str)**: The manufacturer (brand) of the vehicle (default is "FORD").
    """
    years= cfg.main.years_list
    make = cfg.main.make
    path = to_absolute_path(cfg.main.raw_data_path)
    print(years)
    print(type(years))
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
        csv_filename = path
        df.to_csv(csv_filename, index=False, encoding="utf-8")
        print(f"All data saved to: {csv_filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error making the request: {e}")

def check_and_create_env(cfg: DictConfig) -> None:
    """
    - Checks if raw data exists; if not, downloads it.
    - Checks if the conda environment with the name `env_name` exists. 
    - If not, it creates the environment using the `environment.yml` file.

    ###  Parameters:
    - **env_name(str)**: The name of the environment to check/create. Default is 'ford_case'.
    - **path (str)**: The path of the csv file to check/download.
    """
    
    try:
        # Defining the variables 
        file_path = to_absolute_path(cfg.main.raw_data_path)
        env_name = cfg.main.env_name

        #Check if the raw data exists If not download the data
        if not os.path.isfile(file_path):
            download_nhtsa(cfg)


        # Check if the conda environment exists
        result = subprocess.run(['conda', 'info', '--envs'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # If the environment exists, return a message
        print(result)
        print(result.stdout.decode())
        if env_name in result.stdout.decode():
            print(f"Environment '{env_name}' already exists.")
        else:
            print(f"Environment '{env_name}' not found. Creating the environment...")
            # Create the environment from the environment.yml file
            subprocess.run(['conda', 'env', 'create', '-f', 'env.yml'], check=True)
            print(f"Environment '{env_name}' has been created.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while checking/creating the environment: {e}")
        sys.exit(1)
