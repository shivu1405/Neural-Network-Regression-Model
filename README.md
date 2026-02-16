# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement
To build and train a neural network model using PyTorch to predict output values from input data and evaluate its performance using training loss and test data.

## Neural Network Model

Include the neural network model diagram.
<img width="1523" height="887" alt="image" src="https://github.com/user-attachments/assets/e4de192b-2060-4355-93ac-d8ab15e90827" />



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

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = ai_brain(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return losses

losses = train_model(model, X_train, y_train, criterion, optimizer)


model.eval()
with torch.no_grad():
    predictions = model(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nName:Shivasri")
print("Register Number:212224220098")
print("\nEvaluation Results:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# New Sample Prediction
new_sample = np.array([[35,45]])
new_sample_scaled = scaler_X.transform(new_sample)
new_sample_tensor = torch.FloatTensor(new_sample_scaled)

with torch.no_grad():
    pred = model(new_sample_tensor)

pred_actual = scaler_y.inverse_transform(pred.numpy())

print("\nNew Sample Input: [35, 45]")
print("Predicted Output:", pred_actual[0][0])




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
<img width="780" height="225" alt="image" src="https://github.com/user-attachments/assets/2f9bafa2-55a0-4af1-ba73-127f805c247e" />



## RESULT

Include your result here
The program was executed using Command Prompt by placing the dataset and Python file in the same directory.
