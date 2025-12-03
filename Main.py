import os
from gui import launch_and_get_files
from Analyzer import run_batch_analysis

def main():
    """
    Main function to run the HRV analysis application.
    It launches a GUI for specific file selection and then runs 
    the analysis on the selected files.
    """
    print("Welcome to the HRV Analysis tool.")
    
    # Get the directory where this script is located to use as the base for the output folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Launch GUI to get a map of labeled files
    files_map = launch_and_get_files()
    
    # 2. Check if the file map is populated
    if not files_map:
        print("\nNo files were selected for analysis. Exiting program.")
        return
        
    print(f"\nStarting analysis for {len(files_map)} file(s)வுகளில்")
    for label, path in files_map.items():
        print(f"  - {label}: {path}")
        
    # 3. Run the analysis
    run_batch_analysis(files_map, base_dir)

if __name__ == "__main__":
    main()