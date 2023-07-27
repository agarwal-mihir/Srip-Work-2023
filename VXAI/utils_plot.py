import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
from utils import get_simple_data_train

torch.manual_seed(42)
np.random.seed(42)

def plot_generic(x_train, y_train, add_to_plot=None, coef_1=0.3, coef_2=0.02):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xlim([-.5, 1.5])
    plt.ylim([-1.5, 2.5])
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)

    # x_train, y_train = get_simple_data_train(coef_1, coef_2)

    x_true = np.linspace(-.5, 1.5, 1000)
    y_true = x_true + coef_1 * np.sin(2 * np.pi * x_true) + coef_1 * np.sin(4 * np.pi * x_true)

    ax.plot(x_train, y_train, 'ko', markersize=4, label="observations")
    ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")
    if add_to_plot is not None:
        add_to_plot(ax)

    plt.legend(loc=4, fontsize=15, frameon=False)
    return fig, ax

def plot_multiple_predictions(x_train, y_train,x_test, y_preds,coef_1, coef_2):
    def add_multiple_predictions(ax):
        for idx in range(len(y_preds)):
            ax.plot(x_test, y_preds[idx], '-', linewidth=3)

    return plot_generic(x_train, y_train,add_multiple_predictions,coef_1, coef_2)

def plot_predictions(x_train, y_train,x_test, y_preds, coef_1=0.3, coef_2=0.02):
    def add_predictions(ax):
        ax.plot(x_test, y_preds, 'r-', linewidth=3, label='neural net prediction')

    fig, ax = plot_generic(x_train, y_train,add_predictions, coef_1, coef_2)
    return fig, ax

def plot_uncertainty_bands(x_train, y_train,x_test, y_preds,coef_1, coef_2):
    y_preds = np.array(y_preds)
    y_mean = y_preds.mean(axis=0)
    y_std = y_preds.std(axis=0)

    def add_uncertainty(ax):
        ax.plot(x_test, y_mean, '-', linewidth=3, color="#408765", label="predictive mean")
        ax.fill_between(x_test.ravel(), y_mean - 2 * y_std, y_mean + 2 * y_std, alpha=0.6, color='#86cfac', zorder=5)

    return plot_generic(x_train, y_train,add_uncertainty,coef_1, coef_2)