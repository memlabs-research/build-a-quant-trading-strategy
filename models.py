import torch.nn as nn

# Simple Linear Model
class LinearModel(nn.Module):
    def __init__(self, input_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)  # Single output (return prediction)
    
    def forward(self, x):
        return self.linear(x)

# Non-Linear Neural Network Model
class NonLinearModel(nn.Module):
    def __init__(self, input_features, hidden_size=64):
        super(NonLinearModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),  # Non-linear activation
            nn.Linear(hidden_size , 1)  # Output layer
        )
    
    def forward(self, x):
        return self.network(x)