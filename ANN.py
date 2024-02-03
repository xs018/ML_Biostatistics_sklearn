import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Check if GPU is available and set device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. Reading data
data = pd.read_excel("ML_photocatalysis_matlab.xlsx")
x = data.iloc[:, 1:4].values
y = data.iloc[:, 4].values

# Convert data to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
y_tensor = y_tensor.view(y_tensor.shape[0], 1)  # Reshape for PyTorch

# 2. Define a neural network class
class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 3. Training Neural Networks with varying hidden neurons
MaxHiddenNum = 20
RMSE1 = []
Rsqure = []

for s in range(1, MaxHiddenNum + 1):
    model = Net(input_size=x.shape[1], hidden_size=s).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Train the model
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(x_tensor)
        mse = criterion(predictions, y_tensor)
        rmse = torch.sqrt(mse)
        RMSE1.append(rmse.item())
        Rsqure.append(1 - mse.item() / np.var(y))

    # Save model
    torch.save(model.state_dict(), f'net_{s}.pt')

# 4. Analysis and Visualization
plt.plot(range(1, MaxHiddenNum + 1), RMSE1, 'r+')
plt.xlabel('Number of Neurons in Hidden Layer')
plt.ylabel('Root Mean Square Error')
plt.show()

# Finding the best model
best_hidden_size = np.argmin(RMSE1) + 1
print(f"Best hidden layer size: {best_hidden_size}")
