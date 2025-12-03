import os
from gui import select_files
from Analyzer import run_batch_analysis

def main():
    """
    Main function to run the HRV analysis application.
    It opens a GUI for file selection and then runs the analysis
    on the selected files.
    """
    print("Welcome to the HRV Analysis tool.")
    print("Please select one or more ECG CSV files from the dialog window.")
    
    # Get the directory where this script is located to use as the base for the output folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Open GUI to select files
    file_paths = select_files()
    
    # 2. Check if files were selected
    if not file_paths:
        print("\nNo files were selected. Exiting program.")
        return
        
    print(f"\nSelected {len(file_paths)} file(s) for analysis.")

    # 3. Create the files_map from the list of paths
    # The key (label) for each file will be its filename without the extension
    files_map = {}
    for path in file_paths:
        # os.path.basename gets the filename (e.g., 'data.csv')
        # os.path.splitext splits it into ('data', '.csv')
        label = os.path.splitext(os.path.basename(path))[0]
        files_map[label] = path
        
    # 4. Run the analysis
    run_batch_analysis(files_map, base_dir)

if __name__ == "__main__":
    main()
