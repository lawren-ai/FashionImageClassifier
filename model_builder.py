
import torch.nn as nn


# Create a MNIST model by subclassing nn.Module
class MNISTModel(nn.Module):
  def __init__(self, input_shape, hidden_units, output_shape):
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=input_shape, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=output_shape),
        # nn.ReLU()
    )

  # create the fprward pass
  def forward(self, x):
    return self.layer_stack(x)
