import torch
import torch.nn as nn
import torch.optim as optim

# Dummy data and model (example)
X_train = torch.randn(100, 5)  # Example input data
y_train = torch.randint(0, 2, (100,))  # Example labels (binary)


# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Dummy function to count intersecting pairs (placeholder)
def count_intersecting_pairs(model, X, y):
    """
    Dummy function to return a fixed value (10) for counting intersecting pairs.

    Args:
    - model: The neural network model.
    - X: Input data.
    - y: Ground truth labels.

    Returns:
    - 10: Fixed number of intersecting pairs (placeholder).
    """
    return 10


# Initialize model and optimizer
model = SimpleNN(input_size=5, hidden_size=10, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Direct optimization loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    # Compute non-differentiable loss
    loss = count_intersecting_pairs(model, X_train, y_train)

    # Print progress
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss}')

    # Perform direct optimization (adjust model parameters based on loss)
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data -= 0.001 * param.grad  # Adjust parameters manually (not using gradients)

    # Dummy example: Clear gradients for next iteration
    optimizer.zero_grad()
