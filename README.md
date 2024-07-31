# Moodys ESG Data Extractor 

This repository offers an advanced data extraction pipeline utilizing Gemini 1.5 to extract energy consumption data from ESG documents.

## Prerequisites 🛠️

1. **Create Service Credentials on GCP**
2. **Download the JSON Key**

## Setup Steps 📝

### 1. Clone the Repository
```bash
git clone https://github.com/arunpshankar/moodys-esg-data-extractor.git
cd moodys-esg-data-extractor
```

### 2. Create Credentials Folder
- **Create a folder named `credentials`**
- **Drop the JSON key inside and name it `key.json`**

### 3. Update Configuration
- **Navigate to `config/config.yml`**
- **Update the file with your `project_id`, `bucket_name`, and `region`**

### 4. Create a Virtual Environment
```bash
python3 -m venv .moodys-esg-data-extractor
source .moodys-esg-data-extractor/bin/activate
```

### 5. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Set Python Path
```bash
export PYTHONPATH=$PYTHONPATH:.
```

## Data Organization 📂

- **All PDF documents are under `./data/`**
- **Ground truth expected extractions are under `./data/validated/expected`**

## Running Evaluations 🏃‍♂️

### Single Step Extraction
```bash
python src/pipeline/single_step.py
```
*This approach extracts data using a single prompt.*

### Multi-Step Extraction
```bash
python src/pipeline/multi_step.py
```
*This approach extracts data in 4 steps:*
- **Step 0:** Metadata Extraction (Independent)
- **Step 1, 2, 3:** Serial Pipeline Sequence

*The output of the test run is stored in `./data/output` depending on your workflow type (single or multi-step).*

### Validation Extraction
```bash
python src/pipeline/validation/single_step.py
python src/pipeline/validation/multi_step.py
```

- **The `./data/validation/` folder contains extractions by file ID in JSONL format**
- **During the run, JSON files are converted to JSONLs for easy evaluation**

### Evaluation Metrics
- **The `./data/evaluation` folder contains the coverage metric and matched items by file name**