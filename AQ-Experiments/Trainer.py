import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Trainer:
    """
    Trains the model on the given data for the specified number of epochs.

    Args:
        model (nn.Module): The model to train.
        data (torch_geometric.data.Data): The data to train on.
        epochs (int): The number of epochs to train for.
        device (str): The device to train on.

    """

    def __init__(self, model, data, epochs, device):
        self.model = model
        self.data = data
        self.epochs = epochs
        self.device = device

    def train(self, lr):
        """
        Trains the model.

        """
        self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(self.data.x.to(self.device), self.data.edge_index.to(self.device))
            loss = criterion(out[self.data.train_mask].squeeze(), self.data.y[self.data.train_mask].squeeze())
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(self.data.x.to(self.device), self.data.edge_index.to(self.device))  # Predicted PM values
                    train_rmse = torch.sqrt(torch.mean((out[self.data.train_mask].squeeze() - self.data.y[self.data.train_mask].squeeze())**2))
                    test_rmse = torch.sqrt(torch.mean((out[self.data.test_mask].squeeze() - self.data.y[self.data.test_mask].squeeze())**2))
                    print(f"Epoch {epoch}: Train RMSE {train_rmse:.4f}, Test RMSE {test_rmse:.4f}")
                self.model.train()
