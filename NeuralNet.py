"""
By Nathan Bighetti
    Bellow is the code for a neural network with one hidden layer with three nodes and an output layer with one node. 
5/9/2023
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Sigmoid activation function
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Feedforward function
def feedforward(inputs, weights, biases):
    hidden_layer = sigmoid(np.dot(inputs, weights[0]) + biases[0])
    output_layer = np.dot(hidden_layer, weights[1]) + biases[1]
    return output_layer

# Load the input data from an Excel file
data = pd.read_excel('data.xlsx')

# Split the data into input (X) and output (y)
X = data['Year'].values.reshape(-1, 1)
y = data['Tot_Emp'].values.reshape(-1, 1)

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, shuffle=False)

# Normalize the input and output data
X_train = X_train / X_train.max()
y_train = y_train / y_train.max()
X_test = X_test / X_test.max()
y_test = y_test / y_test.max()

# Initialize the weights and biases for the neural network picking at random
weights = [
    np.random.rand(1, 3),
    np.random.rand(3, 1)
]
biases = [
    np.random.rand(1, 3),
    np.random.rand(1, 1)
]

# Traininnig loop using gradient descent
learning_rate = 0.01559
for i in range(1000000):
    # Forward pass
    hidden_layer = sigmoid(np.dot(X_train, weights[0]) + biases[0])
    output_layer = np.dot(hidden_layer, weights[1]) + biases[1]
    
    # Backward pass
    error = y_train - output_layer
    delta_output = error
    delta_hidden = np.dot(delta_output, weights[1].T) * hidden_layer * (1 - hidden_layer)
    
    # Update weights and biases
    weights[1] += learning_rate * np.dot(hidden_layer.T, delta_output)
    biases[1] += learning_rate * np.sum(delta_output, axis=0, keepdims=True)
    weights[0] += learning_rate * np.dot(X_train.T, delta_hidden)
    biases[0] += learning_rate * np.sum(delta_hidden, axis=0)

# Predict the number of employees for the years 2020 and 2021
X_test_pred = X_test
y_test_pred = feedforward(X_test_pred, weights, biases) * y_train.max()

# Print the accuracy of the predicted values for 2020 and 2021
print("Predicted values for 2020 and 2021:")
print(y_test_pred)
#Print the actual values predicted for 2020 and 2021
print("Actual predicted values for testing data:")
print(y_test_pred / y_train.max() * data['Tot_Emp'].max())

