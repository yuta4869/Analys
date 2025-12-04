import os

from Analyzer import run_batch_analysis
from GenerateBoxPlots import generate_box_plots
from gui import launch_ui


def main():
    print("Welcome to the HRV Analysis tool.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    combined_file_path = os.path.join(base_dir, "result_batch", "Combined_HRV_Analysis.xlsx")

    def run_callback(files_map):
        print(f"\nStarting analysis for {len(files_map)} file(s)")
        for label, path in files_map.items():
            print(f"  - {label}: {path}")
        run_batch_analysis(files_map, base_dir)

    def boxplot_callback():
        return generate_box_plots(combined_file_path, os.path.dirname(combined_file_path))

    launch_ui(run_callback, boxplot_callback)


if __name__ == "__main__":
    main()
