### DISCLAIMER: 
- #### The CSV files in data/ are truncated samples (10 rows) for demonstration. For full reproduction, download the CADEC v2 dataset [[CADEC_V2](https://data.csiro.au/collection/csiro:10948v3)] and replace these files.
- #### To Run the code successfully, a Google GEMINI API Key is required.

#### Data Availability:

##### This project utilizes the CADEC v2 dataset. Due to size constraints, the full dataset is not included in this repository.

- Download: [[CADEC_V2](https://data.csiro.au/collection/csiro:10948v3)]
- Setup Steps: Keep the directory structure as is.
    - Retain the dir structure
    - Unzip the data and keep the `original` and `text` folder inside `data/cadec_v2/`
    - Run the script inside `scripts/preprocess_cadec_v2_data.py`
    - Result: A csv with file name `cadec_v2.csv` will be created in `data/cadec_v2/`
- Sample: the directory `data/cadec_v2/` have sample values and should be replaced by the original data downloaded from above url and following the steps mentioned.