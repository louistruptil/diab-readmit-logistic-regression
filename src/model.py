import torch
import torch.nn as nn

class LogisticRegression(nn.Module):

    def __init__(self, in_features: int):

        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=1, bias=True)

    def forward(self, x:torch.tensor) -> torch.Tensor:
        z = self.linear(x)
        y_prob = torch.sigmoid(z)
        return y_prob