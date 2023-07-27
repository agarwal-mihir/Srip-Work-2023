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

def display_equation(coef_1, coef_2):
    equation = r"f(x, \varepsilon) = x + {:.2f} \sin(2 \pi (x + \varepsilon)) + {:.2f} \sin(4 \pi (x + \varepsilon)) + \varepsilon".format(coef_1, coef_1)
    st.header("Custom Function")
    st.latex(equation)

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