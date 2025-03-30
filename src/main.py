import tkinter as tk
import sys
import os

# Add the parent directory to sys.path to enable absolute imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.gui.main_window import MainWindow

def main():
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()