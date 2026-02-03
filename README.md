# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement
To build and train a neural network model using PyTorch to predict output values from input data and evaluate its performance using training loss and test data.

## Neural Network Model

Include the neural network model diagram.
<img width="1033" height="682" alt="image" src="https://github.com/user-attachments/assets/626f3235-9f33-4652-803d-87b730281859" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Shivasri.S
### Register Number: 212224220098
```python


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("dataset.csv")

X = data.iloc[:, :-1].values   # Input features
y = data.iloc[:, -1].values.reshape(-1, 1)  # Output

-
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_model(model, X_train, y_train, criterion, optimizer, epochs=2000):
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.6f}")

    return losses

losses = train_model(model, X_train, y_train, criterion, optimizer)


plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Training Loss Vs Iteration")
plt.show()

with torch.no_grad():
    test_output = model(X_test)
    test_loss = criterion(test_output, y_test)
    print("Testing Loss:", test_loss.item())

sample_input = torch.tensor([[60, 70]], dtype=torch.float32)
sample_input_scaled = scaler_x.transform(sample_input)
sample_input_scaled = torch.tensor(sample_input_scaled, dtype=torch.float32)

predicted_scaled = model(sample_input_scaled)
predicted = scaler_y.inverse_transform(predicted_scaled.detach().numpy())

print("Sample Input: [60, 70]")
print("Predicted Output:", predicted[0][0])



```
## Dataset Information

Include screenshot of the dataset
<img width="1919" height="1129" alt="image" src="https://github.com/user-attachments/assets/d46db56f-bab7-4897-862a-3eb3d10d2d3e" />


## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot here
<img width="792" height="595" alt="image" src="https://github.com/user-attachments/assets/7b64a460-c1ed-4727-bc1a-c20458620fb0" />


### New Sample Data Prediction

Include your sample input and output here
<img width="1487" height="594" alt="image" src="https://github.com/user-attachments/assets/3ec9a442-59cd-41d5-a462-336bdd4b0da4" />


## RESULT

Include your result here
The program was executed using Command Prompt by placing the dataset and Python file in the same directory.
