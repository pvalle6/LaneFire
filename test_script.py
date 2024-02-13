import tkinter as tk

class TestClass:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("")
        self.canvas = tk.Canvas(self.root)
        self.canvas.grid()
    def run(self):
        self.root.mainloop()
TestClass().run()
