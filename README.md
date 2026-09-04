<H3>Name</H3> Sanjana J 
<H3>Register no.</H3> 212224230240
<H3>Date</H3> 04-09-26
<H3>Experiment No. 2 </H3>
## Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)

# -------- Graph 1: 3D Iris dataset --------
X3 = df.iloc[:, 0:3].values

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X3[:50,0], X3[:50,1], X3[:50,2], label="Setosa")
ax.scatter(X3[50:100,0], X3[50:100,1], X3[50:100,2], label="Versicolour")
ax.scatter(X3[100:150,0], X3[100:150,1], X3[100:150,2], label="Virginica")
ax.set_xlabel("Sepal length")
ax.set_ylabel("Sepal width")
ax.set_zlabel("Petal length")
ax.legend()
plt.show()

# -------- Take 2 classes --------
X = df.iloc[:100, [0,1]].values
y = np.where(df.iloc[:100,4] == "Iris-setosa", 1, -1)

# -------- Graph 2: 2D data --------
plt.scatter(X[:50,0], X[:50,1], label="Setosa")
plt.scatter(X[50:,0], X[50:,1], marker="x", label="Versicolour")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()
plt.show()

# Scale data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# -------- Perceptron --------
w = np.zeros(2)
b = 0
lr = 0.01
errors = []

for epoch in range(10):
    error = 0

    for xi, yi in zip(X_train, y_train):
        pred = 1 if np.dot(xi, w) + b >= 0 else -1
        update = lr * (yi - pred)

        w += update * xi
        b += update

        if update != 0:
            error += 1

    errors.append(error)

# Predict
pred = np.where(np.dot(X_test, w) + b >= 0, 1, -1)

print("Accuracy:", accuracy_score(y_test, pred) * 100, "%")

# -------- Graph 3: Error --------
plt.plot(range(1, 11), errors, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.title("Perceptron Training Error")
plt.show()
```
    

# OUTPUT:

   <img width="616" height="512" alt="image" src="https://github.com/user-attachments/assets/e90628ec-8e56-4c6c-b0e5-d48a93217b25" />

<img width="750" height="542" alt="image" src="https://github.com/user-attachments/assets/930230fa-cd98-4b33-86ac-dc7cfe676b5f" />

<img width="750" height="601" alt="image" src="https://github.com/user-attachments/assets/ff83933c-8308-45cf-9707-fdf4e05698ae" />


# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
