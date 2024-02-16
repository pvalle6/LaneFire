"""
    This is the GUI interface file for LaneFire.

    Peter Vallet 2024
"""

import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
import pickle as pkl
import LaneFire
import pandas as pd
import re


class LaneFireGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LaneFire")
        self.root.geometry("")
        self.root.attributes('-topmost', True)

        """
            Building UI for Application
        """
        self.current_experiment = None
        self.cleaned_input_data = pd.DataFrame()

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

        self.start_button = tk.Button(self.splash, text="Start", command=self.start_lanefire).grid()

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

        self.start_select_act = tk.Button(self.start_q_canvas, text="Begin", command=self.start_run_type)
        self.start_select_act.pack()

    #
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

        self.plot_d_button = tk.Button(self.evaluate_exp, text="Plot Data", command=self.plot_data)
        self.plot_d_button.grid()
        self.ask_exp = tk.Button(self.evaluate_exp, text="Ask Experiment", command=self.run_old_bofire)
        self.ask_exp.grid()
        self.save_exp = tk.Button(self.evaluate_exp, text="Save Experiment to PKL", command=self.save_pickle)
        self.save_exp.grid()
        self.table_header = tk.Label(self.evaluate_exp, text="Experiments Provided").grid()

        """
        New Experiment
        """
        self.new_exp = tk.Canvas(self.root)
        self.new_exp_header = tk.Label(self.new_exp, text="New Experiment Creation Wizard").grid()
        self.starting_data_path = None

        self.get_data_button = tk.Button(self.new_exp,
                                         text="Choose a starting data file", command=self.start_new_exp).grid()
        self.use_same_data_button = tk.Button(self.new_exp,
                                              text="Use last opened data", command=self.load_same_data).grid()

        self.new_exp_error = tk.Label(self.new_exp, text="You must choose a file with data to continue!")

        """
        Data Cleaning Window
        """
        self.Data_Cleaning = tk.Canvas(self.root)
        self.ld_st_data_header = tk.Label(self.Data_Cleaning, text="Data Loaded Successfully!").grid()
        self.st_data_txt = "ERROR"

        self.loaded_start_data_display = tk.Label(self.Data_Cleaning, text=self.st_data_txt)
        self.data_cleaning_label = tk.Label(
            self.Data_Cleaning, text="Data Cleaning Window \n See instructions README.txt for format").grid()

        self.dc_var_range_col = tk.StringVar()
        self.dc_instr_var_col = tk.Label(self.Data_Cleaning, text="Insert Column Range for Variables (X)").grid()
        self.dc_var_col = tk.Entry(self.Data_Cleaning, textvariable=self.dc_var_range_col).grid()

        self.dc_range_row = tk.StringVar()
        self.dc_instr_var_row = tk.Label(self.Data_Cleaning, text="Insert Row Range for Variables (X)").grid()
        self.dc_var_row = tk.Entry(self.Data_Cleaning, textvariable=self.dc_range_row).grid()

        self.dc_obj_range_col = tk.StringVar()
        self.dc_instr_obj_col = tk.Label(self.Data_Cleaning, text="Insert Column Range for Objectives (Y)").grid()
        self.dc_obj_col = tk.Entry(self.Data_Cleaning, textvariable=self.dc_obj_range_col).grid()

        self.dc_instr_obj_row = tk.Label(self.Data_Cleaning, text="Insert Row Range for Objectives (Y)").grid()
        self.dc_obj_row = tk.Entry(self.Data_Cleaning, textvariable=self.dc_range_row).grid()

        self.cleaned_display = tk.Label(self.Data_Cleaning, text="")
        self.dc_parameters_button = tk.Button(self.Data_Cleaning, text="Save Ranges", command=self.save_dc_param).grid()
        self.original_data_label = tk.Label(self.Data_Cleaning, text="Original Data").grid()

        self.cleaned_label = tk.Label(self.Data_Cleaning, text="Cleaned Data")
        self.len_var = 0
        self.len_obj = 0

        self.use_clean_data = tk.Button(self.Data_Cleaning, text="Use Cleaned Data", command=self.begin_setup)

        """
        Set BOFIRE Parameters 
        """

        self.setup_bofire_canvas = tk.Canvas(self.root)
        self.setup_bofire_label = tk.Label(self.setup_bofire_canvas, text="BOFIRE Settings").grid(row=0)

        self.data_description = "Number of Variables: {0} \n Number of Objectives: {1}".format(
            self.len_var, self.len_obj-self.len_obj)
        self.setbo_description = tk.Label(self.setup_bofire_canvas, text=self.data_description)


        self.set_var_iter = 0
        self.weight_holder = []
        self.bound_holder = []
        self.obj_type = []
        self.target_list = []

        self.type_var = tk.IntVar()
        self.type_rb1 = tk.Radiobutton(self.setup_bofire_canvas, text="Maximize", variable=self.type_var, value=1)
        self.type_rb2 = tk.Radiobutton(self.setup_bofire_canvas, text="Minimize", variable=self.type_var, value=2)
        self.type_rb3 = tk.Radiobutton(self.setup_bofire_canvas, text="Target", variable=self.type_var, value=3)
        self.target_var = tk.StringVar()
        self.target_label = tk.Label(self.setup_bofire_canvas, text = "Obj Target")
        self.type_target_e = tk.Entry(self.setup_bofire_canvas, textvariable=self.target_var)

        self.weights_label = tk.Label(self.setup_bofire_canvas, text="Enter Weights").grid(row=3)
        self.weight_var = tk.StringVar(value="0")
        self.weights_var_entry = tk.Entry(self.setup_bofire_canvas, textvariable=self.weight_var).grid(row=4)

        self.up_bounds_var = tk.StringVar(value="0")
        self.low_bounds_var = tk.StringVar(value="0")

        self.up_bounds_label = tk.Label(self.setup_bofire_canvas, text="Enter Upper Bounds").grid(row=5)
        self.up_bounds_var_entry = tk.Entry(self.setup_bofire_canvas, textvariable=self.up_bounds_var).grid(row=6)

        self.low_bounds_label = tk.Label(self.setup_bofire_canvas, text="Enter Lower Bounds").grid(row=7)
        self.low_bounds_var_entry = tk.Entry(self.setup_bofire_canvas, textvariable=self.low_bounds_var).grid(row=8)

        self.set_settings_button = tk.Button(self.setup_bofire_canvas, text="Set Settings", command=self.get_settings)
        self.set_settings_button.grid(row=15)

        """
        Canvas to Verify all Bofire Settings 
        """
        self.new_domain = None
        self.verify_bofire = tk.Canvas(self.root)
        self.verify_header = tk.Label(self.verify_bofire, text="Verify the Following Settings for BoFire").grid()
        self.verify_table = str()
        self.verify_description = tk.Label(self.verify_bofire, text=self.verify_table)
        self.setup_bofire_domain_button = tk.Button(self.verify_bofire, text="Set Up Bofire Strategy",
                                                    command=self.gen_domain).grid()

        """
        Ask Bofire Screen
        """
        self.ask_bofire_screen = tk.Canvas(self.root)
        self.run_bf_scr_label = tk.Label(self.ask_bofire_screen, text="Bofire Run Screen").grid()

        self.run_bf_scr_asks = tk.Label(self.ask_bofire_screen,
                                        text="How Many Asks? " +
                                             "(More than 1 Will Result in Very High Computational Demand").grid()
        self.asks_wheel = tk.Spinbox(self.ask_bofire_screen, from_=1, to=5)
        self.asks_wheel.grid()
        self.run_bf_button = tk.Button(self.ask_bofire_screen,
                                       text="Ask Bofire for next experiment", command=self.run_new_bofire).grid()

        """
        Running Bofire Screen
        """
        self.bofire_ask_scr = tk.Canvas(self.root)
        self.bf_ask_header = tk.Label(self.bofire_ask_scr, text="Bofire Results").grid()
        self.bf_ask_header_two = tk.Label(self.bofire_ask_scr, text="Informed Candidates").grid()
        self.bf_ask_results = tk.Label(self.bofire_ask_scr)
        self.save_exp_new = tk.Button(self.bofire_ask_scr, text="Save Experiment to PKL", command=self.save_pickle)
        self.plot_new = tk.Button(self.bofire_ask_scr, text="Plot Experiment Suggestion", command=self.plot_data)

    def start_lanefire(self):
        self.splash.grid_forget()
        self.start_q_canvas.grid()

    def start_run_type(self):
        self.start_q_canvas.grid_forget()

        if self.start_choice.get() == 0:
            self.new_exp.grid()
        elif self.start_choice.get() == 1:
            self.load_exp.grid()  # opens screen to open previous experiment

    def load_same_data(self):
        """
        Uses the same path previously used for loading data
        :return:
        """
        self.new_exp.grid_forget()

        with open(r".\Experiment_Data\.previous_run_path.txt", "r") as file:
            old_data_path = file.read()

        self.new_data_df = pd.read_csv(str(old_data_path))

        self.Data_Cleaning.grid()
        self.st_data_txt = self.new_data_df.to_string()
        self.loaded_start_data_display = tk.Label(self.Data_Cleaning, text=self.st_data_txt, relief=GROOVE).grid()

    def start_new_exp(self):
        """
        Start a new experiments a new experiment run in BoFire

        Need to implement dimensionality variables

        :return:
        """
        starting_data_path = fd.askopenfilename()


        if self.starting_data_path != "":
            self.new_exp.grid_forget()
            self.new_data_df = pd.read_csv(str(starting_data_path))

            with open(r".\Experiment_Data\.previous_run_path.txt", "w") as file:
                file.write(str(starting_data_path))

            self.Data_Cleaning.grid()
            self.st_data_txt = self.new_data_df.to_string()
            self.loaded_start_data_display = tk.Label(self.Data_Cleaning, text=self.st_data_txt, relief=GROOVE).grid()

        else:
            self.new_exp_error.grid()

    def save_dc_param(self):
        self.cleaned_display.grid()
        self.cleaned_display.grid_forget()

        """
        Saves the parameters set in the Data Cleaning Window for use of Data!

        self.dc_var_range_col
        self.dc_range_row
        self.dc_obj_range_col

        the parameters here have to be very specific;

        :return:
        """

        """
        this isn't a very robust implementation , but it is original 
        
        to get a single row/col, use n_i; eg. "1"
        
        to get a single range, use n_0:n_m inclusively ; eg. "4:6"
        
        to use discontinuous rows, use n_0, n_m; eg. "7,8,9"
        
        to chain any of these together, separate them by open and close parentheses
        eg. (7,8)(9:21)
        """
        dc_param = dict()
        dc_param.update({"dc_var_range_col": self.dc_var_range_col.get()})
        dc_param.update({"dc_range_row": self.dc_range_row.get()})
        dc_param.update({"dc_obj_range_col": self.dc_obj_range_col.get()})
        self.cleaned_input_data = self.clean_data(dc_param)
        self.cleaned_label.grid()
        self.cleaned_display = tk.Label(self.Data_Cleaning, text=str(self.cleaned_input_data), relief=GROOVE).grid()
        self.use_clean_data.grid()
        self.data_description = "Number of Variables: {0} \n Number of Objectives: {1}".format(
            self.len_var, self.len_obj-self.len_var)
        print(self.data_description)

    def clean_data(self, dc_input_param):
        """
        This constructs a pd df using the parameters set in the parameters.
        :return:
        """
        # self.new_data_df has original
        cleaned_df = pd.DataFrame()
        """
        The header implementation of the cleaned data set needs to be overhauled
        """

        def parse_param(input_string):
            """
            REGEX CODE PARTIALLY WRITTEN BY GPT-3.5

            :param input_string:
            :return:
            """
            # Use regular expression to find substrings within parentheses
            regex_pattern = r'\((.*?)\)|([^()]+)'
            matches = re.findall(regex_pattern, input_string)

            # Filter out empty strings and return the result
            return [match[0] if match[0] else match[1] for match in matches]

        obj_col_input = parse_param(dc_input_param.get("dc_obj_range_col"))
        row_input = parse_param(dc_input_param.get("dc_range_row"))
        var_col_input_input = parse_param(dc_input_param.get("dc_var_range_col"))

        """
        Below is the most disgusting use of pandas slicing every seen, I apologize to anyone who views this. 
        It is mostly pythonic but by no means is inline with pandas style guide.
        """

        for param in row_input:
            row_input_hold = list()
            if "," in param and ":" in param:
                print("ERROR MSG: Mixed Use of Parameter Ranges")
            elif ":" in param:
                row_begin, row_end = param.split(":")
                for i in range(int(row_begin), int(row_end)):
                    row_input_hold.append(i)
            elif "," in param:
                row_input_hold = param.split(",")
                row_input_hold = [int(x) for x in row_input_hold]
            else:
                row_input_hold = [int(param)]

        list_col_index = list()
        for param in var_col_input_input:
            var_col_input_hold = list()
            if "," in param and ":" in param:
                print("ERROR MSG: Mixed Use of Parameter Ranges")
            elif ":" in param:
                var_col_begin, var_col_end = param.split(":")
                for i in range(int(var_col_begin), int(var_col_end)):
                    var_col_input_hold.append(i)
            elif "," in param:
                var_col_input_hold = param.split(",")
                var_col_input_hold = [int(x) for x in var_col_input_hold]
            else:
                var_col_input_hold = [int(param)]
            for rows in row_input_hold:
                for col in var_col_input_hold:
                    cleaned_df.at[rows, col] = self.new_data_df.iloc[rows, col]
            for col in var_col_input_hold:
                list_col_index.append(col)
            self.len_var = len(list_col_index)

        for param in obj_col_input:
            obj_col_input_hold = list()
            if "," in param and ":" in param:
                print("ERROR MSG: Mixed Use of Parameter Ranges")
            elif ":" in param:
                obj_col_begin, obj_col_end = param.split(":")
                for i in range(int(obj_col_begin), int(obj_col_end)):
                    obj_col_input_hold.append(i)
            elif "," in param:
                obj_col_input_hold = (param.split(","))
                obj_col_input_hold = [int(x) for x in obj_col_input_hold]
            else:
                obj_col_input_hold = [int(param)]
            for rows in row_input_hold:
                for col in obj_col_input_hold:
                    cleaned_df.at[rows, col] = self.new_data_df.iloc[rows, col]
            for col in obj_col_input_hold:
                list_col_index.append(col)
        self.len_obj = len(list_col_index)
        col_iter = 0
        for col_index in list_col_index:
            cleaned_df.rename({cleaned_df.columns[col_iter]: self.new_data_df.columns[col_index]},
                              inplace=True, axis=1)
            print({cleaned_df.columns[col_iter]: self.new_data_df.columns[col_index]})
            col_iter = col_iter + 1

        print(cleaned_df)
        return cleaned_df

    def get_settings(self):
        self.weight_holder.append(float(self.weight_var.get()))
        self.bound_holder.append((float(self.low_bounds_var.get()), float(self.up_bounds_var.get())))
        self.set_var_iter = self.set_var_iter + 1

        """
        This needs to be refactored into appending an Experiment instance rather than a global variable
        """

        if self.set_var_iter > self.len_var:
            self.obj_type.append(int(self.type_var.get()))
            if int(self.type_var.get()) == 3:
                self.target_list.append(float(self.target_var.get()))
            else:
                self.target_list.append(None)
        if self.set_var_iter >= self.len_var:
            self.type_rb1.grid(row=9)
            self.type_rb2.grid(row=10)
            self.type_rb3.grid(row=11)
            self.target_label.grid(row=12)
            self.type_target_e.grid(row=13)

        else:
            self.type_rb1.grid_forget()
            self.type_rb2.grid_forget()
            self.type_rb3.grid_forget()

        if self.set_var_iter < len(self.cleaned_input_data.columns):
            self.var_label.grid_forget()
            self.setting_var_label = self.cleaned_input_data.columns[self.set_var_iter]
            self.var_label = tk.Label(self.setup_bofire_canvas, text=self.setting_var_label)
            self.var_label.grid(row=1)
        else:
            self.verification_gen()
            self.verify_bofire.grid()
            self.verify_description = tk.Label(self.verify_bofire, text=self.verify_table).grid()
            self.setup_bofire_canvas.grid_forget()

    def begin_setup(self):
        """
        Sets up the bofire settings
        :return:
        """
        self.Data_Cleaning.grid_forget()
        self.setup_bofire_canvas.grid()

        self.setting_var_label = self.cleaned_input_data.columns[self.set_var_iter]
        self.var_label = tk.Label(self.setup_bofire_canvas, text=self.setting_var_label)
        self.var_label.grid(row=1)
        self.setbo_description = tk.Label(self.setup_bofire_canvas, text=self.data_description).grid()

        """
        This will be creating a LaneFireParam instance to be fed into the BofirePipelineSetup.
        """
    def verification_gen(self):
        self.verify_table = "EXP/OBJ | WEIGHT | BOUND \n"
        for index, header in enumerate(self.cleaned_input_data.columns):
            self.verify_table = (self.verify_table + str(header) + ":" +
                                 str(self.weight_holder[index]) + ":" + str(self.bound_holder[index]) + "\n")

    def gen_domain(self):
        """
        Generates a domain for the strategy used in Bofire.

        :return:
        """

        """
        This needs to be implemented before and its properties need to be created along with the experiment.
        """
        self.verify_bofire.grid_forget()
        self.ask_bofire_screen.grid()
        new_param = LaneFire.LaneFireParam(
            var_names=self.cleaned_input_data.columns[0:self.len_var],
            obj_names=self.cleaned_input_data.columns[self.len_var:self.len_obj],
            var_bound_tuples=self.bound_holder[0:self.len_var],
            obj_bound_tuples=self.bound_holder[self.len_var:self.len_obj],
            list_var_weights=self.weight_holder[0:self.len_var],
            list_obj_weights=self.weight_holder[self.len_var:self.len_obj],
            list_opt_types=self.obj_type,
            obj_targets=self.target_list
        )
        print(new_param.list_obj_weights)
        self.new_domain = LaneFire.bofire_setup_pipe(new_param)

    def run_new_bofire(self):
        self.ask_bofire_screen.grid_forget()
        self.bofire_ask_scr.grid()
        self.candidates = LaneFire.bofire_ask_update(self.new_domain, self.cleaned_input_data,
                                                     int(self.asks_wheel.get()))
        self.bf_ask_results = tk.Label(self.bofire_ask_scr, text=self.candidates).grid()

        """
        Need to refactor this during the new experiment screen
        """
        self.current_experiment = LaneFire.Experiment()
        self.current_experiment.original_provided_exp = self.cleaned_input_data
        self.current_experiment.list_predictions.append(self.candidates)
        self.current_experiment.var_n = self.len_var
        self.current_experiment.obj_n = self.len_obj - self.len_var
        self.current_experiment.domain = self.new_domain
        self.save_exp_new.grid()
        self.plot_new.grid()

    def run_old_bofire(self):
        self.bofire_ask_scr.grid()
        self.candidates = LaneFire.bofire_ask_update(self.current_experiment.domain,
                                                     self.current_experiment.original_provided_exp,
                                                     int(self.asks_wheel.get()))
        self.bf_ask_results = tk.Label(self.bofire_ask_scr, text=self.candidates).grid()
        self.current_experiment.list_predictions.append(self.candidates)
        self.save_exp_new.grid()
        # self.plot_new.grid()

    def find_pickle(self):
        """
        Find a pickle file for a previous experiment run and loads it.
        :return:
        """
        loaded_pkl_exp = fd.askopenfilename()

        if loaded_pkl_exp != "":
            self.load_exp.grid_forget()

            with open(loaded_pkl_exp, "rb") as handle:
                self.current_experiment = pkl.load(handle, encoding='UTF-8')

            self.experiment_history = tk.Label(self.evaluate_exp,
                                               text=self.current_experiment.original_provided_exp)
            self.experiment_history.grid()
            self.evaluate_exp.grid()
        else:
            self.load_pkl_error.grid()

    def save_pickle(self):
        """
        Saves an experiment run to be used later
        :return:
        """
        sfile = fd.asksaveasfilename()

        with open(sfile, "wb") as handle:
            pkl.dump(self.current_experiment, handle)

    def plot_data(self):
        """
        Plots an experiment run with candidates

        needs to implement scaling
        :return:
        """
        LaneFire.plot_candidates(self.current_experiment)

    # def plot_with(self):
    #     """
    #     Plots an experiment with deviations along side provided data
    #     :return:
    #     """
    #     LaneFire.plot_clean(self.current_experiment.original_provided_exp,
    #     self.current_experiment.list_predictions[0])

    def run(self):
        self.root.mainloop()


LaneFireGUI().run()
