import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
from utils import get_simple_data_train, display_equation, train
from utils_plot import plot_generic, plot_multiple_predictions, plot_predictions, plot_uncertainty_bands
from model import MLP

torch.manual_seed(42)
np.random.seed(42)

def main():
    st.title("Conformal Prediction: A Brief Overview")

    st.subheader("Introduction:")
    st.write("Machine learning models, especially black-box models like neural networks, have gained widespread adoption in high-risk domains like medical diagnostics, where accurate predictions are critical to avoid potential model failures. However, the lack of uncertainty quantification in these models poses significant challenges in decision-making and trust. Conformal prediction emerges as a promising solution, providing a user-friendly paradigm to quantify uncertainty in model predictions.")

    st.write("The key advantage of conformal prediction lies in its distribution-free nature, making it robust to various scenarios without making strong assumptions about the underlying data distribution or the model itself. By utilizing conformal prediction with any pre-trained model, researchers and practitioners can create prediction sets that offer explicit, non-asymptotic guarantees, instilling confidence in the reliability of model predictions.")


    st.write("Conformal Prediction is a powerful framework in machine learning that provides a measure of uncertainty in predictions. Unlike traditional point predictions, conformal prediction constructs prediction intervals that quantify the range of potential outcomes.")

    st.write("The significance of conformal prediction lies in its ability to provide a confidence level (alpha) for the predictions, allowing users to understand the reliability of the model's output. This is especially crucial in critical applications where understanding the uncertainty is essential.")

    
    coef_1 = st.slider("Coefficient of function", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    coef_2 = st.slider("Coefficient for noise", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    x, y = get_simple_data_train(coef_1, coef_2)
    cal_idx = np.arange(len(x), step=1/0.2, dtype=np.int64)
    # cal_idx = np.random.choice(len(x), size=int(len(x) * 0.2), replace=False) # random selection
    mask = np.zeros(len(x), dtype=bool)
    mask[cal_idx] = True
    x_cal, y_cal = x[mask], y[mask]
    x_train, y_train = x[~mask], y[~mask]
    st.title("Custom Function Visualization")
    display_equation(coef_1, coef_2)

    st.header("Data Visualization")
    fig, ax = plot_generic(x_train, y_train,coef_1=coef_1, coef_2=coef_2)
    st.pyplot(fig)

    # Train the model
    x_test = torch.linspace(-.5, 1.5, 3000)[:, None] 
    net = MLP(hidden_dim=30, n_hidden_layers=2)
    net = train(net, (x_train, y_train))

    # Make predictions and plot the results
    y_preds = net(x_test).clone().detach().numpy()


    # Display the plot with the true function and observations
    st.header("Prediction Visualization")
    fig, ax = plot_predictions(x_train, y_train,x_test, y_preds,coef_1, coef_2)
    st.pyplot(fig)

    ensemble_size = 5
    ensemble = [MLP(hidden_dim=30, n_hidden_layers=2) for _ in range(ensemble_size)]
    for net in ensemble:
        train(net, (x_train, y_train))
    y_preds = [np.array(net(x_test).clone().detach().numpy()) for net in ensemble]
    fig, ax = plot_multiple_predictions(x_train, y_train,x_test, y_preds,coef_1, coef_2)
    st.pyplot(fig)
    fig, ax = plot_uncertainty_bands(x_train, y_train,x_test, y_preds,coef_1, coef_2)
    st.pyplot(fig)

    net_dropout = MLP(hidden_dim=30, n_hidden_layers=2, use_dropout=True)
    net_dropout = train(net_dropout, (x_train, y_train))
    n_dropout_samples = 100

    # compute predictions, resampling dropout mask for each forward pass
    y_preds = [net_dropout(x_test).clone().detach().numpy() for _ in range(n_dropout_samples)]
    y_preds = np.array(y_preds)
    fig, ax = plot_multiple_predictions(x_train, y_train,x_test, y_preds,coef_1, coef_2)
    st.pyplot(fig)
    fig, ax = plot_uncertainty_bands(x_train, y_train,x_test, y_preds,coef_1, coef_2)
    st.pyplot(fig)
    x_test = torch.linspace(-.5, 1.5, 1000)[:, None]
    y_preds = net(x_test).clone().detach().numpy()
    y_cal_preds = net(x_cal).clone().detach()
    resid = torch.abs(y_cal - y_cal_preds).numpy()
    alpha = 0.1
    n = len(x_cal)
    q_val = np.ceil((1 - alpha) * (n + 1)) / n
    q = np.quantile(resid, q_val, method="higher")
    x_true = np.linspace(-.5, 1.5, 1000)
    y_true = x_true + 0.3 * np.sin(2 * np.pi * x_true) + 0.3 * np.sin(4 * np.pi * x_true)

    # generate plot
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xlim([-.5, 1.5])
    plt.ylim([-1.5, 2.5])
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)

    ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")
    ax.plot(x, y, 'ko', markersize=4, label="observations")
    ax.plot(x_test, y_preds, '-', linewidth=3, color="#408765", label="predictive mean")
    ax.fill_between(x_test.ravel(), y_preds - q, y_preds + q, alpha=0.6, color='#86cfac', zorder=5)

    plt.legend(loc=4, fontsize=15, frameon=False);
    st.pyplot(fig)
if __name__ == "__main__":
    main()