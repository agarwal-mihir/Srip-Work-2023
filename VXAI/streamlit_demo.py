import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm

torch.manual_seed(42)
np.random.seed(42)

def get_simple_data_train(coef_1, coef_2):
    x = np.linspace(-.2, 0.2, 500)
    x = np.hstack([x, np.linspace(.6, 1, 500)])
    eps = coef_2 * np.random.randn(x.shape[0])
    y = x + coef_1 * np.sin(2 * np.pi * (x + eps)) + coef_1 * np.sin(4 * np.pi * (x + eps)) + eps
    x_train = torch.from_numpy(x).float()[:, None]
    y_train = torch.from_numpy(y).float()
    return x_train, y_train

def plot_generic(add_to_plot=None, coef_1=0.3, coef_2=0.02):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xlim([-.5, 1.5])
    plt.ylim([-1.5, 2.5])
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)

    x_train, y_train = get_simple_data_train(coef_1, coef_2)

    x_true = np.linspace(-.5, 1.5, 1000)
    y_true = x_true + coef_1 * np.sin(2 * np.pi * x_true) + coef_1 * np.sin(4 * np.pi * x_true)

    ax.plot(x_train, y_train, 'ko', markersize=4, label="observations")
    ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")
    if add_to_plot is not None:
        add_to_plot(ax)

    plt.legend(loc=4, fontsize=15, frameon=False)
    return fig, ax

def display_equation(coef_1, coef_2):
    equation = r"f(x, \varepsilon) = x + {:.2f} \sin(2 \pi (x + \varepsilon)) + {:.2f} \sin(4 \pi (x + \varepsilon)) + \varepsilon".format(coef_1, coef_1)
    st.header("Custom Function")
    st.latex(equation)

class MLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=10, n_hidden_layers=1, use_dropout=False):
        super().__init__()

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.Tanh()

        # dynamically define architecture
        self.layer_sizes = [input_dim] + n_hidden_layers * [hidden_dim] + [output_dim]
        layer_list = [nn.Linear(self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                      range(1, len(self.layer_sizes))]
        self.layers = nn.ModuleList(layer_list)

    def forward(self, input):
        hidden = self.activation(self.layers[0](input))
        for layer in self.layers[1:-1]:
            hidden_temp = self.activation(layer(hidden))

            if self.use_dropout:
                hidden_temp = self.dropout(hidden_temp)

            hidden = hidden_temp + hidden  # residual connection

        output_mean = self.layers[-1](hidden).squeeze()
        return output_mean

def train(net, train_data):
    x_train, y_train = train_data
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    progress_bar = trange(3000)
    for _ in progress_bar:
        optimizer.zero_grad()
        loss = criterion(y_train, net(x_train))
        progress_bar.set_postfix(loss=f'{loss / x_train.shape[0]:.3f}')
        loss.backward()
        optimizer.step()
    return net

def plot_predictions(x_test, y_preds, coef_1=0.3, coef_2=0.02):
    def add_predictions(ax):
        ax.plot(x_test, y_preds, 'r-', linewidth=3, label='neural net prediction')

    fig, ax = plot_generic(add_predictions, coef_1, coef_2)
    return fig, ax

def main():
    st.title("Simple MLP Regression with Streamlit")
    coef_1 = st.slider("Coefficient 1", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    coef_2 = st.slider("Coefficient 2", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    x_train, y_train = get_simple_data_train(coef_1,coef_2)
    st.title("Custom Function Visualization")
    display_equation(coef_1, coef_2)

    st.header("Data Visualization")
    fig, ax = plot_generic(coef_1=coef_1, coef_2=coef_2)
    st.pyplot(fig)

    # Train the model
    x_test = torch.linspace(-.5, 1.5, 3000)[:, None] 
    net = MLP(hidden_dim=30, n_hidden_layers=2)
    net = train(net, (x_train, y_train))

    # Make predictions and plot the results
    y_preds = net(x_test).clone().detach().numpy()


    # Display the plot with the true function and observations
    st.header("Prediction Visualization")
    fig, ax = plot_predictions(x_test, y_preds,coef_1, coef_2)
    st.pyplot(fig)

if __name__ == "__main__":
    main()