import os.path

from src import download_data, utils

path = 'data/raw/full_data_2020_2025_FORD.csv'

if not os.path.isfile(path):
    download_data.download_nhtsa(years= list(range(2020, 2026)))

utils.check_and_create_env()
