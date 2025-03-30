import tkinter as tk
from tkinter import ttk

class ResultsView:
    def __init__(self, parent):
        self.parent = parent
        
        # Create a frame for the text widget
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        self.text = tk.Text(self.frame, wrap=tk.WORD, state=tk.DISABLED)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text.config(yscrollcommand=self.scrollbar.set)
        
    def add_message(self, message):
        """Add a new message to the results view."""
        self.text.configure(state=tk.NORMAL)
        if self.text.index('end-1c') != '1.0':
            self.text.insert(tk.END, '\n')
        self.text.insert(tk.END, message)
        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)
        
    def clear(self):
        """Clear all messages from the results view."""
        self.text.configure(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)
        self.text.configure(state=tk.DISABLED)
        
    def get_all_text(self):
        """Get all text from the results view."""
        return self.text.get(1.0, tk.END)