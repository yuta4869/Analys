# HRV Analysis from ECG Data

This project provides Python scripts to analyze Heart Rate Variability (HRV) from ECG data stored in CSV files. It processes the raw ECG signal to calculate R-R intervals (RRI) and subsequently derives frequency-domain (LF/HF) and time-domain (RMSSD) HRV metrics.

## Main Entrypoint

The main entry point for the application is `Main.py`, which launches a graphical interface for file selection.

## Features

-   **GUI File Selection:** Easily select one or more analysis files through a user-friendly dialog.
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

This application can be run in two ways:

### 1. Using the GUI (Recommended)

The main entry point is `Main.py`, which provides a graphical user interface to select your input files.

1.  **Run the Main Script:** Execute `Main.py` from your terminal:

    ```bash
    python Main.py
    ```

2.  **Select Files:** A file dialog will appear. Select one or more ECG CSV files that you want to analyze.

3.  **Check Results:** The analysis will run automatically, and the output Excel files will be saved in a `result_batch` directory within the project folder.

### 2. Using the Command Line (Advanced)

You can still run the analysis by modifying the `Analyzer.py` script directly. This is useful for environments without a graphical interface.

1.  **Configure Input Files:** Open `Analyzer.py` and modify the `default_files_map` dictionary with the absolute paths to your files.

2.  **Run the Script:**

    ```bash
    python Analyzer.py
    ```
