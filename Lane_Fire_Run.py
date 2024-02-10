"""
    This is the GUI interface file for LaneFire.

    Peter Vallet 2024
"""

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
import LaneFire
from LaneFire import Experiment
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


class LaneFireGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LaneFire")
        self.root.geometry("")
        # self.root.resizable(False, False)
        self.root.attributes('-topmost', True)

        """
            Building UI for Application
        """
        self.current_experiment = None

        """
        Splash is the opening screen
        """
        self.splash = tk.Canvas(self.root)
        wlkom_msg = ("Welcome to LaneFire \n \n LaneFire is an open source GUI interface for BoFire " +
                     "\nFor help see {LaneFireRepo}" +
                     "\n \n Peter Vallet 2024" +
                     "\nLinkedIn: https://www.linkedin.com/in/peter-v-334609211/" +
                     "\nGitHub: https://github.com/pvalle6 \n \n")

        self.splash.grid(column=0, row=0)
        self.welcome_msg = tk.Label(self.splash, text=wlkom_msg).grid()

        self.start_button = tk.Button(self.splash, text="Start", command=self.start_question_de).grid()

        """
        The window for selecting an option for new experiment or loading old one
        """
        self.start_q_canvas = tk.Canvas(self.root)
        self.question = tk.Label(self.start_q_canvas, text="Start from New or Load Experiment?\n")
        self.question.pack()

        self.start_choice = tk.IntVar()

        self.start_rb_1 = tk.Radiobutton(self.start_q_canvas, text="New Experiment", variable=self.start_choice,
                                         value=0)
        self.start_rb_2 = tk.Radiobutton(self.start_q_canvas, text="Load Experiment", variable=self.start_choice,
                                         value=1)
        self.start_rb_1.pack()
        self.start_rb_2.pack()

        self.start_select_act = tk.Button(self.start_q_canvas, text="Begin", command=self.start_select_do)
        self.start_select_act.pack()

        """
        Load Experiment Window
        """
        self.load_exp = tk.Canvas(self.root)
        self.load_exp_header = tk.Label(self.load_exp, text="Choose a Previous Experiment's Pickle File\n").grid()
        self.loaded_pkl_exp = str()
        self.select_path = tk.Button(self.load_exp, text="Find File", command=self.find_pickle).grid()
        self.load_pkl_error = tk.Label(self.load_exp, text="PLEASE SELECT A FILE BEFORE CONTINUING")

        """
        Evaluate Experiment
        """
        self.evaluate_exp = tk.Canvas(self.root)
        self.loaded_exp = tk.Label(self.evaluate_exp, text="Experiment Loaded!").grid()

        self.plot_exp = tk.Button(self.evaluate_exp, text="Plot Experiments", command=self.plot_exp)
        self.plot_exp.grid()
        self.ask_exp = tk.Button(self.evaluate_exp, text="Ask Experiment", command=self.ask_exp)
        self.ask_exp.grid()
        self.save_exp = tk.Button(self.evaluate_exp, text="Save Experiment to PKL", command=self.save_pickle)
        self.save_exp.grid()
        self.table_header = tk.Label(self.evaluate_exp, text="Experiments Provided").grid()

    def start_question_de(self):
        self.splash.grid_forget()
        self.start_q_canvas.grid()

    def start_select_do(self):
        self.start_q_canvas.grid_forget()
        if self.start_choice.get() == 0:
            pass
        if self.start_choice.get() == 1:
            self.load_exp.grid()

    def find_pickle(self):
        self.loaded_pkl_exp = fd.askopenfilename()
        if self.loaded_pkl_exp != "":
            self.load_exp.grid_forget()

            with open(self.loaded_pkl_exp, "rb") as handle:
                self.current_experiment = pkl.load(handle, encoding='UTF-8')

            self.experiment_history = tk.Label(self.evaluate_exp,
                                               text=self.current_experiment.original_provided_exp)
            self.experiment_history.grid()
            self.evaluate_exp.grid()
        else:
            self.load_pkl_error.grid()

    def save_pickle(self):
        sfile = fd.asksaveasfile()
        
        with open(sfile, "wb") as handle:
            pkl.dump(self.current_experiment, handle, encoding='UTF-8')

    def run(self):
        self.root.mainloop()

    def plot_exp(self):
        LaneFire.print_wo(self.current_experiment.original_provided_exp)

    def ask_exp(self):
        domain = LaneFire.bofire_setup_pipe()
        LaneFire.bofire_ask_update(domain, self.current_experiment.original_provided_exp, 1)


LaneFireGUI().run()
