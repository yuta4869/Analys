import tkinter as tk
from tkinter import filedialog

def select_files():
    """
    Opens a file dialog to select one or more CSV files.
    
    :return: A list of strings, where each string is the full path to a selected file.
             Returns an empty list if no files are selected.
    """
    # Create a root window but hide it
    root = tk.Tk()
    root.withdraw()
    
    # Define file types
    file_types = [
        ('CSV files', '*.csv'),
        ('All files', '*.*')
    ]
    
    # Open the file dialog to select multiple files
    # The askopenfilenames() function returns a tuple of selected file paths
    selected_files = filedialog.askopenfilenames(
        title='Select ECG CSV files for analysis',
        filetypes=file_types
    )
    
    # The return value is a tuple, convert it to a list
    return list(selected_files)

if __name__ == '__main__':
    # Example of how to use the function
    print("Opening file selection dialog...")
    files = select_files()
    
    if files:
        print("\nYou selected the following files:")
        for file_path in files:
            print(file_path)
    else:
        print("\nNo files were selected.")
