import torch
import torch.nn as nn
import torch.nn.functional as F

# Define XORModel as before
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden = nn.Linear(2, 2)  # Input size: 2, Hidden size: 2
        self.output = nn.Linear(2, 1)  # Hidden size: 2, Output size: 1

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # Apply Sigmoid to Hidden Layer
        x = torch.sigmoid(self.output(x))  # Apply Sigmoid to Output Layer
        return x

# Initialize model, criterion, optimizer
model = XORModel()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# XOR dataset
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Train the model
for epoch in range(5000):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Get weights and biases
state_dict = model.state_dict()
print("Trained Weights and Biases:")
for key, value in state_dict.items():
    print(f"{key}: {value}")

# Extract weights and biases
W1 = state_dict['hidden.weight']  # Weights for hidden layer
b1 = state_dict['hidden.bias']    # Biases for hidden layer
W2 = state_dict['output.weight']  # Weights for output layer
b2 = state_dict['output.bias']    # Biases for output layer

# Perform manual calculations for a specific input
X = torch.tensor([1, 0], dtype=torch.float32)  # Example input

# Step 1: Hidden layer transformation
hidden_output = torch.matmul(X, W1.T) + b1
hidden_activation = torch.sigmoid(hidden_output)

# Step 2: Output layer transformation
output = torch.matmul(hidden_activation, W2.T) + b2
final_output = torch.sigmoid(output)

print("\nManual Calculation Steps:")
print(f"Input: {X}")
print(f"Hidden Layer Output (before activation): {hidden_output}")
print(f"Hidden Layer Activation: {hidden_activation}")
print(f"Output Layer Output (before activation): {output}")
print(f"Final Output (after activation): {final_output}")

# Compare with model's prediction
model.eval()
with torch.no_grad():
    predicted = model(X)
    print(f"\nModel Predicted Output: {predicted}")
