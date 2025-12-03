import tkinter as tk
from tkinter import filedialog, messagebox

class AnalysisGUI:
    def __init__(self, master):
        self.master = master
        master.title("HRV Analysis File Selector")
        master.geometry("500x200")

        self.file_paths = {}
        self.labels = ["Fixed", "Sin", "HRF"]
        self.entries = {}

        # Create input rows
        for i, label_text in enumerate(self.labels):
            label = tk.Label(master, text=f"{label_text}:")
            label.grid(row=i, column=0, padx=10, pady=10, sticky="w")

            entry = tk.Entry(master, width=50)
            entry.grid(row=i, column=1, padx=10, pady=10)
            self.entries[label_text] = entry

            browse_button = tk.Button(master, text="Browse...", command=lambda l=label_text: self.browse_file(l))
            browse_button.grid(row=i, column=2, padx=10, pady=10)

        # Create Run button
        run_button = tk.Button(master, text="Run Analysis", command=self.run_analysis)
        run_button.grid(row=len(self.labels), column=1, pady=20)

    def browse_file(self, label):
        file_path = filedialog.askopenfilename(
            title=f"Select file for {label}",
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if file_path:
            self.entries[label].delete(0, tk.END)
            self.entries[label].insert(0, file_path)
            self.file_paths[label] = file_path

    def run_analysis(self):
        # Final check of file paths from entries before closing
        final_paths = {}
        for label, entry in self.entries.items():
            path = entry.get()
            if path:
                final_paths[label] = path
        
        if not final_paths:
            messagebox.showwarning("No Files", "No files were selected. The program will exit.")
        
        self.file_paths = final_paths
        self.master.quit() # Use quit() to end the mainloop and allow return

def launch_and_get_files():
    """
    Launches the GUI, waits for the user to select files and click 'Run Analysis',
    and returns the selected file paths.

    :return: A dictionary mapping labels to file paths, e.g., {'Fixed': 'path/to/file.csv'}.
             Returns an empty dictionary if no files are selected.
    """
    root = tk.Tk()
    app = AnalysisGUI(root)
    root.mainloop() # This blocks until master.quit() is called
    
    # After mainloop ends, destroy the window and return the paths
    selected_files = app.file_paths
    root.destroy()
    return selected_files


if __name__ == '__main__':
    # Example of how to use the function
    print("Launching GUI to select files...")
    file_map = launch_and_get_files()

    if file_map:
        print("\nAnalysis would run with the following files:")
        for label, path in file_map.items():
            print(f"  {label}: {path}")
    else:
        print("\nNo files were selected or the window was closed.")