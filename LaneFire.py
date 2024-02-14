"""
    LaneFire is an open source GUI interface for the BoFire (https://github.com/experimental-design/bofire).

    Peter Vallet 2024
    LinkedIn: https://www.linkedin.com/in/peter-v-334609211/
    GitHub: https://github.com/pvalle6
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bofire
import pickle as pkl

from bofire.data_models.strategies.api import MoboStrategy as DataModel
from bofire.data_models.features.api import ContinuousInput, DiscreteInput, CategoricalInput, CategoricalDescriptorInput
from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective, TargetObjective
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.constraints.api import LinearEqualityConstraint, LinearInequalityConstraint
from bofire.data_models.domain.api import Constraints
from bofire.data_models.domain.api import Domain
import bofire.strategies.api as strategies


class Experiment:
    """
    This class is responsible for saving and loading experimental data for use in the LaneFire Pipeline. It
    can be saved as a pickle file and loaded from a pickle file.

    It allows for use of LaneFire in memory limited systems.
    """
    def __init__(self):
        """
        Initialises the experiment_class.
        """
        self.x_sample_space = None
        self.y_sample_space = None
        self.original_provided_exp = None
        self.list_predictions = []
        self.provided_charts = None  # This will be implemented later

        self.exp_info = {"Name": None,
                         "Data Points Originally Provided": None, "Number of Additional Samples Added": None}
        self.var_n = None
        self.obj_n = None
        self.domain = None

    def print_info(self):
        """
        converts self.exp_info to printable format and prints/returns it
        :return:
        """
        pass


def plot_clean(provided_exp, informed_candidates):
    pass
    # """
    # This function plots the cleaned data with proposed experimental data including a SD bar around each.
    #
    # It currently is programmed for 1 var and 5 obj. I will work on updating this.
    #
    # :param provided_exp: previously provided experimental data to BoFire
    # :param informed_candidates: predicted experimental data by BoFire
    # """
    # combined_predictions = pd.concat([provided_exp, informed_candidates], ignore_index=True)
    #
    # f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharex=True, clear=True, sharey=True)
    #
    # f.set_figheight(5)
    # f.set_figwidth(15)
    #
    # ax1.errorbar(combined_predictions.loc[:, "x1"], combined_predictions.loc[:, 'y1'],
    #              combined_predictions.loc[:, 'y1_sd'], linestyle='None', marker='^')
    # ax1.set_xlim([0, 1])
    # ax2.errorbar(combined_predictions.loc[:, "x1"], combined_predictions.loc[:, 'y2'],
    #              combined_predictions.loc[:, 'y3_sd'], linestyle='None', marker='^')
    # ax2.set_xlim([0, 1])
    # ax3.errorbar(combined_predictions.loc[:, "x1"], combined_predictions.loc[:, 'y3'],
    #              combined_predictions.loc[:, 'y3_sd'], linestyle='None', marker='^')
    # ax3.set_xlim([0, 1])
    # ax4.errorbar(combined_predictions.loc[:, "x1"], combined_predictions.loc[:, 'y4'],
    #              combined_predictions.loc[:, 'y4_sd'], linestyle='None', marker='^')
    # ax4.set_xlim([0, 1])
    # ax5.errorbar(combined_predictions.loc[:, "x1"], combined_predictions.loc[:, 'y5'],
    #              combined_predictions.loc[:, 'y5_sd'], linestyle='None', marker='^')
    # ax5.set_xlim([0, 1])
    #
    # plt.xlabel("x")
    # plt.show()


def rename_info(informed_candidates):
    """
    Rename the columns of a pandas df to allow for the combined plotting of BoFire predicted experimental data
    and verified experimental data.
    :param informed_candidates:  A pandas df of predicted experimental runs.
    :return renamed_informed: renamed pandas df of predicted experimental runs
    """
    predictions = informed_candidates
    renamed_informed = predictions.rename(columns={"y1_pred": "y1", "y2_pred": "y2"})
    return renamed_informed


def find_nearest(x_np, value):
    """
    Finds the nearest data (x) point to a proposed experimental run.
    :param x_np: A numpy array of single independent variables.
    :param value: The predicted independent variable of a proposed experimental run.
    :return array[idx]: The nearest point to the proposed independent variable.
    """
    array = np.asarray(x_np)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def simulate_experiment(x_np, y_np, given_x):
    """
    Simulates an experimental test of a proposed experimental run by grabbing the data point from a given population.
    :param x_np: A numpy array of x values of a population.
    :param y_np: A numpy array of y values of a population.
    :param given_x: A proposed x value for an experiment.
    :return simulation: A pandas df of simulated experimental results.
    """
    nearest_value = find_nearest(x_np, value=given_x)
    next_exper_index = np.where(x_np == nearest_value)
    simulation = pd.DataFrame({'x1': x_np[next_exper_index], 'y1': y_np[next_exper_index[0][0], 0],
                               'y2': y_np[next_exper_index[0][0], 1], 'y3': y_np[next_exper_index[0][0], 2],
                               'y4': y_np[next_exper_index[0][0], 3], 'y5': y_np[next_exper_index[0][0], 4]})
    return simulation


def print_wo(combined_predictions):
    """
    Prints out the experimental runs when no proposed experiments are involved.
    :param combined_predictions:
    """
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharex=True, clear=True, sharey=True)

    f.set_figheight(5)
    f.set_figwidth(15)

    ax1.errorbar(combined_predictions.loc[:, "x1"], combined_predictions.loc[:, 'y1'], 0, linestyle='None', marker='^')
    ax2.errorbar(combined_predictions.loc[:, "x1"], combined_predictions.loc[:, 'y2'], 0, linestyle='None', marker='^')
    ax3.errorbar(combined_predictions.loc[:, "x1"], combined_predictions.loc[:, 'y3'], 0, linestyle='None', marker='^')
    ax4.errorbar(combined_predictions.loc[:, "x1"], combined_predictions.loc[:, 'y4'], 0, linestyle='None', marker='^')
    ax5.errorbar(combined_predictions.loc[:, "x1"], combined_predictions.loc[:, 'y5'], 0, linestyle='None', marker='^')

    plt.xlabel("x")
    plt.show()


class LaneFireParam:
    def __init__(self, var_names, obj_names, var_bound_tuples, obj_bound_tuples, list_var_weights, list_obj_weights,
                 list_opt_types, obj_targets):
        self.var_n = var_names
        self.obj_n = obj_names
        self.obj_bound_tuples = obj_bound_tuples
        self.var_bound_tuples = var_bound_tuples
        self.list_var_weights = list_var_weights
        self.list_obj_weights = list_obj_weights
        self.list_opt_types = list_opt_types
        self.obj_targets = obj_targets


def bofire_setup_pipe(LaneFireParam):
    """
    A function contained code based in and modified from the BoFire Repo. This preps any use of the bofire package and
    use of ask or tell functions.

    Needs to be expanded to be modified easily.
    :return:
    """

    """
    This is also a butchered setup, there is definitely a better way to implement this. 
    """

    x1 = ContinuousInput(key=LaneFireParam.var_n[0], bounds=LaneFireParam.var_bound_tuples[0])
    input_features = Inputs(features=[x1])

    if len(LaneFireParam.var_bound_tuples) >= 2:
        x2 = ContinuousInput(key=LaneFireParam.var_n[1], bounds=LaneFireParam.var_bound_tuples[1])
    if len(LaneFireParam.var_bound_tuples) >= 3:
        x3 = ContinuousInput(key=LaneFireParam.var_n[2], bounds=LaneFireParam.var_bound_tuples[2])
        input_features = Inputs(features=[x1, x2, x3])
    if len(LaneFireParam.var_bound_tuples) >= 4:
        x4 = ContinuousInput(key=LaneFireParam.var_n[3], bounds=LaneFireParam.var_bound_tuples[3])
        input_features = Inputs(features=[x1, x2, x3, x4])
    if len(LaneFireParam.var_bound_tuples) >= 5:
        x5 = ContinuousInput(key=LaneFireParam.var_n[4], bounds=LaneFireParam.var_bound_tuples[4])
        input_features = Inputs(features=[x1, x2, x3, x4, x5])
    if len(LaneFireParam.var_bound_tuples) >= 6:
        print("TOO MANY INDEPENDENT VARIABLES; HIGHER DIMENSIONALITY NOT YET IMPLEMENTED")

    """
    No further variables are included as I don't think there is a situation that needs this much
    
    There's gotta be a more pythonic way to do this.
    """

    """
    Here the program checks the user given parameters to implement 1. the number of objectives, and the type of 
    optimization desired for each objective. 
    """

    if len(LaneFireParam.obj_bound_tuples) >= 1:
        if LaneFireParam.list_opt_types[0] == 1:
            objective1 = MaximizeObjective(
                w=LaneFireParam.list_obj_weights[0],
                bounds=LaneFireParam.obj_bound_tuples[0])
            y1 = ContinuousOutput(key=LaneFireParam.obj_n[0], objective=objective1)
        elif LaneFireParam.list_opt_types[0] == 2:
            objective1 = MinimizeObjective(
                w=LaneFireParam.list_obj_weights[0],
                bounds=LaneFireParam.obj_bound_tuples[0])
            y1 = ContinuousOutput(key=LaneFireParam.obj_n[0], objective=objective1)
        elif LaneFireParam.list_opt_types[0] == 3:
            objective1 = TargetObjective(
                w=LaneFireParam.list_obj_weights[0],
                bounds=LaneFireParam.obj_bound_tuples[0],
                target=LaneFireParam.obj_targets[0])
            y1 = ContinuousOutput(key=LaneFireParam.obj_n[0], objective=objective1)

        output_features = Outputs(features=[y1])

    if len(LaneFireParam.obj_bound_tuples) >= 2:
        if LaneFireParam.list_opt_types[1] == 1:
            objective2 = MaximizeObjective(
                w=LaneFireParam.list_obj_weights[1],
                bounds=LaneFireParam.obj_bound_tuples[1])
            y2 = ContinuousOutput(key=LaneFireParam.obj_n[1], objective=objective2)
        elif LaneFireParam.list_opt_types[1] == 2:
            objective2 = MinimizeObjective(
                w=LaneFireParam.list_obj_weights[1],
                bounds=LaneFireParam.obj_bound_tuples[1])
            y2 = ContinuousOutput(key=LaneFireParam.obj_n[1], objective=objective2)
        elif LaneFireParam.list_opt_types[1] == 3:
            objective2 = TargetObjective(
                w=LaneFireParam.list_obj_weights[1],
                bounds=LaneFireParam.obj_bound_tuples[1],
                target=LaneFireParam.obj_targets[1])
            y2 = ContinuousOutput(key=LaneFireParam.obj_n[1], objective=objective2)

        output_features = Outputs(features=[y1, y2])
    if len(LaneFireParam.obj_bound_tuples) >= 3:
        if LaneFireParam.list_opt_types[2] == 1:
            objective3 = MaximizeObjective(
                w=LaneFireParam.list_obj_weights[2],
                bounds=LaneFireParam.obj_bound_tuples[2])
            y3 = ContinuousOutput(key=LaneFireParam.obj_n[2], objective=objective3)
        elif LaneFireParam.list_opt_types[2] == 2:
            objective3 = MinimizeObjective(
                w=LaneFireParam.list_obj_weights[2],
                bounds=LaneFireParam.obj_bound_tuples[2])
            y3 = ContinuousOutput(key=LaneFireParam.obj_n[2], objective=objective3)
        elif LaneFireParam.list_opt_types[0] == 3:
            objective3 = TargetObjective(
                w=LaneFireParam.list_obj_weights[2],
                bounds=LaneFireParam.obj_bound_tuples[2],
                target=LaneFireParam.obj_targets[2])
            y3 = ContinuousOutput(key=LaneFireParam.obj_n[2], objective=objective3)

        output_features = Outputs(features=[y1, y2, y3])
    if len(LaneFireParam.obj_bound_tuples) >= 4:
        if LaneFireParam.list_opt_types[3] == 1:
            objective4 = MaximizeObjective(
                w=LaneFireParam.list_obj_weights[3],
                bounds=LaneFireParam.obj_bound_tuples[3])
            y4 = ContinuousOutput(key=LaneFireParam.obj_n[3], objective=objective4)
        elif LaneFireParam.list_opt_types[3] == 2:
            objective4 = MinimizeObjective(
                w=LaneFireParam.list_obj_weights[3],
                bounds=LaneFireParam.obj_bound_tuples[3])
            y4 = ContinuousOutput(key=LaneFireParam.obj_n[3], objective=objective4)
        elif LaneFireParam.list_opt_types[3] == 3:
            objective4 = TargetObjective(
                w=LaneFireParam.list_obj_weights[3],
                bounds=LaneFireParam.obj_bound_tuples[3],
                target=LaneFireParam.obj_targets[3])
            y4 = ContinuousOutput(key=LaneFireParam.obj_n[3], objective=objective4)
        output_features = Outputs(features=[y1, y2, y3, y4])
    if len(LaneFireParam.obj_bound_tuples) >= 5:
        if LaneFireParam.list_opt_types[4] == 1:
            objective5 = MaximizeObjective(
                w=LaneFireParam.list_obj_weights[4],
                bounds=LaneFireParam.obj_bound_tuples[4])
            y5 = ContinuousOutput(key=LaneFireParam.obj_n[4], objective=objective5)
        elif LaneFireParam.list_opt_types[4] == 2:
            objective5 = MinimizeObjective(
                w=LaneFireParam.list_obj_weights[4],
                bounds=LaneFireParam.obj_bound_tuples[4])
            y5 = ContinuousOutput(key=LaneFireParam.obj_n[4], objective=objective5)
        elif LaneFireParam.list_opt_types[4] == 3:
            objective5 = TargetObjective(
                w=LaneFireParam.list_obj_weights[4],
                bounds=LaneFireParam.obj_bound_tuples[4],
                target=LaneFireParam.obj_targets[4])
            y5 = ContinuousOutput(key=LaneFireParam.obj_n[4], objective=objective5)
        output_features = Outputs(features=[y1, y2, y3, y4, y5])

    if len(LaneFireParam.obj_bound_tuples) >= 6:
        print("TOO MANY OBJECTIVES; HIGHER DIMENSIONALITY NOT YET IMPLEMENTED")
    """
       No further objectives are included as I don't think there is a situation that needs this much
    """

    # A mixture: x1 + x2 + x3 = 1
    # constr1 = LinearEqualityConstraint(features=["x1", "x2", "x3"], coefficients=[1,1,1], rhs=1)

    # x1 + 2 * x3 < 0.8
    # constr2 = LinearInequalityConstraint(features=["x1", "x3"], coefficients=[1, 2], rhs=0.8)

    constraints = []
    domain = Domain(inputs=input_features, outputs=output_features, constraints=constraints)

    return domain


def bofire_ask_update(domain, original_exp, ask_n):
    strategy_data_model = DataModel(domain=domain)
    strategy_data_model = strategies.map(strategy_data_model)
    strategy_data_model.tell(experiments=original_exp)
    informed_candidates = strategy_data_model.ask(ask_n)
    return informed_candidates
