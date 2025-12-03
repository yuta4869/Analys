# HRV Analysis from ECG Data

This project provides Python scripts to analyze Heart Rate Variability (HRV) from ECG data stored in CSV files. It processes the raw ECG signal to calculate R-R intervals (RRI) and subsequently derives frequency-domain (LF/HF) and time-domain (RMSSD) HRV metrics.

## Main Script

The primary script for analysis is `Analyzer.py`.

## Features

-   **ECG Processing:** Reads ECG data from CSV files and applies a bandpass filter.
-   **R-Peak Detection:** Identifies R-peaks in the ECG signal to calculate RRI.
-   **HRV Metrics:**
    -   Calculates the LF/HF ratio (frequency-domain).
    -   Calculates RMSSD (time-domain).
-   **Sliding Window Analysis:** Performs analysis over a sliding window to observe changes over time.
-   **Excel Output:** Saves the analysis results, including time-series data and overall metrics, into Excel files.

## Installation

The script requires several Python libraries. You can install them using pip:

```bash
pip install numpy pandas matplotlib scipy openpyxl
```

## Usage

1.  **Configure Input Files:** Open the `Analyzer.py` script and modify the `files_map` dictionary to specify the labels and absolute paths of the CSV files you wish to analyze.

    ```python
    files_map = {
        "Label1": "/path/to/your/ecg_data_1.csv",
        "Label2": "/path/to/your/ecg_data_2.csv",
    }
    ```

2.  **Run the Script:** Execute the script from your terminal:

    ```bash
    python Analyzer.py
    ```

3.  **Check Results:** The output Excel files will be saved in a `result_batch` directory within the project folder.
