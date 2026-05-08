# Some general FAQ on ML-basics

## Write a simple regression model using Keras

Here is an example of a simple neural network regression model using Keras:
```python
# Import necessary modules
import numpy as np
from tensorflow import keras

# Define the model
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=1))

# Compile the model
model.compile(optimizer="sgd", loss="mean_squared_error")

# Generate dummy data
x = np.linspace(-1, 1, 100)
y = x * x

# Fit the model to the data
model.fit(x, y, epochs=50)

# Use the model to make predictions
predictions = model.predict(x)

# Print the predictions
print(predictions)
```
In this example, we first import the necessary modules, including numpy and keras from TensorFlow. We then define a simple Keras model using the Sequential class. This model has a single dense layer with one unit and an input dimension of one.

Next, we compile the model using the compile() method. We specify the "sgd" optimizer and the "mean_squared_error" loss function.

We then generate some dummy data using numpy.linspace(). This data consists of a set of x-values and the corresponding y-values (which are equal to the squares of the x-values).

Next, we use the fit() method to train the model on the dummy data. We specify the number of epochs (i.e. iterations over the data) to be 50.

Finally, we use the predict() method to make predictions using the trained model. We print the resulting predictions to the console.

When you run this code, the model will be trained on the dummy data, and the predictions made by the model will be printed to the console. These predictions should be close to the true y-values, since the model has learned to approximate the square function from the training data.

## Using a custom loss function in Keras

*An adaption of this is needed when modifying the loss function.*

Here is an example of a neural network regression model in Keras that uses a custom loss function:
```python
# Import necessary modules
import numpy as np
from tensorflow import keras

# Define the custom loss function
def my_loss_fn(y_true, y_pred):
    # Compute the mean squared error
    mse = keras.losses.mean_squared_error(y_true, y_pred)
    # Return the mean squared error multiplied by 2
    return mse * 2

# Define the model
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=1))

# Compile the model with the custom loss function
model.compile(optimizer="sgd", loss=my_loss_fn)

# Generate dummy data
x = np.linspace(-1, 1, 100)
y = x * x

# Fit the model to the data
model.fit(x, y, epochs=50)

# Use the model to make predictions
predictions = model.predict(x)

# Print the predictions
print(predictions)
```
In this example, we first define a custom loss function called my_loss_fn(). This function takes the true y-values and the predicted y-values as arguments, and computes the mean squared error between them using the mean_squared_error() function from Keras. The function then returns the mean squared error multiplied by 2.

We then define a simple Keras model using the Sequential class. This model has a single dense layer with one unit and an input dimension of one.

Next, we compile the model using the compile() method. We specify the "sgd" optimizer and our custom loss function, my_loss_fn(), as the loss function.

We then generate some dummy data using numpy.linspace(). This data consists of a set of x-values and the corresponding y-values (which are equal to the squares of the x-values).

Next, we use the fit() method to train the model on the dummy data. We specify the number of epochs (i.e. iterations over the data) to be 50.

Finally, we use the predict() method to make predictions using the trained model. We print the resulting predictions to the console.

When you run this code, the model will be trained on the dummy data using the custom loss function, and the predictions made by the model will be printed to the console. These predictions should be close to the true y-values, since the model has learned to approximate the square function from the training data. Note that the predictions will be less accurate than they would be if we were using the default mean squared error loss function, since our custom loss function multiplies the mean squared error by 2, which makes it more difficult for the model to learn the correct mapping from x-values to y-values.

## Using separate inputs in a neural network using keras

Here is an example of a Keras neural network regression model that uses separate input layers for different input features:
```python
# Import necessary modules
import numpy as np
from tensorflow import keras

# Define the model
input_a = keras.Input(shape=(1,))
input_b = keras.Input(shape=(1,))

output_a = keras.layers.Dense(units=1)(input_a)
output_b = keras.layers.Dense(units=1)(input_b)

concat = keras.layers.concatenate([output_a, output_b])
output = keras.layers.Dense(units=1)(concat)

model = keras.Model(inputs=[input_a, input_b], outputs=output)

# Compile the model
model.compile(optimizer="sgd", loss="mean_squared_error")

# Generate dummy data
x1 = np.linspace(-1, 1, 100)
x2 = x1 ** 2
y = x1 * x2

# Fit the model to the data
model.fit([x1, x2], y, epochs=50)

# Use the model to make predictions
predictions = model.predict([x1, x2])

# Print the predictions
print(predictions)
```
In this example, we first define a Keras model that uses separate input layers for two different input features. We create two Input layers, one for each input feature, and specify the shape of each input (in this case, a single value). We then define two dense layers, one for each input, and connect each input to its corresponding dense layer.

Next, we concatenate the outputs of the two dense layers using the concatenate layer from Keras. We then define a final dense layer that takes the concatenated outputs as input and produces the final output of the model.

We then create a Model instance using the Model class from Keras, and specify the input layers and output layers that we defined earlier.

Next, we compile the model using the compile() method and specify the "sgd" optimizer and the "mean_squared_error" loss function.

We then generate some dummy data using numpy.linspace(). This data consists of two sets of x-values (x1 and x2) and the corresponding y-values (which are equal to the product of x1 and x2).

Next, we use the fit() method to train the model on the dummy data. We specify the number of epochs (i.e. iterations over the data) to be 50.

Finally, we use the predict() method to make predictions using the trained model. We print the resulting predictions to the console.

When you run this code, the model will be trained on the dummy data, and the predictions made by the model will be printed to the console. These predictions should be close to the true y-values, since the model has learned to approximate the function that maps x1 and x2 to y from the training data.

## What is difference between a CNN and Dense neural network?
A convolutional neural network (CNN) is a type of neural network that is specifically designed to process data that has a grid-like structure, such as an image. A CNN typically consists of multiple layers, including convolutional layers, pooling layers, and fully-connected (dense) layers.

A dense neural network, on the other hand, is a type of neural network that is composed of only fully-connected (dense) layers. This type of network is not specifically designed to process grid-like data, and can be applied to a wider range of data types and problem domains.

The main difference between a CNN and a dense neural network is the architecture of the network. A CNN is designed to exploit the spatial structure of the data, while a dense neural network is a more general-purpose model that can be applied to many different types of data. This makes CNNs well-suited to tasks such as image classification and object detection, while dense neural networks are better suited to tasks such as regression and natural language processing.

## How do I split a dataset into training, validation, and test set using python?

To split a dataset into training, validation, and test sets in Python, you can use the train_test_split() function from the scikit-learn library. This function takes the input dataset as a numpy array and splits it into three subsets: a training set, a validation set, and a test set.

Here is an example of how you might use the train_test_split() function to split a dataset into training, validation, and test sets:
```python
# Import the train_test_split() function
from sklearn.model_selection import train_test_split

# Load the input data
X = ... # numpy array of shape (num_samples, num_features)
y = ... # numpy array of shape (num_samples,)

# Split the data into training, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, test_size=0.1, val_size=0.1)
```
In this example, we first import the train_test_split() function from scikit-learn. We then load the input data as a numpy array, where X contains the input features and y contains the corresponding labels.

Next, we use the train_test_split() function to split the input data into three subsets: a training set, a validation set, and a test set. We specify the size of the test and validation sets using the test_size and val_size parameters, respectively. The function returns six arrays containing the input and output data for each subset.
