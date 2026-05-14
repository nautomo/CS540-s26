# add your code to this file
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv #u can use this to compute the inverse of a matrix

# Globals
x = None
y = None
xmin = None
xmax = None
X_tilde = None
theta_closed = None
losses = None

"""
Plot year vs. number of frozen days and save it to "data_plot.jpg" using plt.savefig.
"""
def data_visualization():
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.savefig("data_plot.jpg")
    plt.close()

"""
Print the normalized and augmented polynomial feature matrix eX. Use the following format for your print statements:
    print("Q2:")
    print(X_tilde)
"""
def data_normalization(m):
    global xmin, xmax, X_tilde
    xmin = np.min(x)
    xmax = np.max(x)
    x_norm = (x - xmin) / (xmax - xmin)

    n = len(x_norm)
    X_tilde = np.zeros((n, m + 1))

    for i in range(m):
        X_tilde[:, i] = x_norm ** (i + 1)

    X_tilde[:, m] = 1
    
    print("Q2:")
    print(X_tilde)

"""
Print the optimal parameter vector θ = (w1, . . . , wm, b) as a numpy array. Use the following format:
    print("Q3:")
    print(theta)
"""
def closed_poly_regression():
    global theta_closed
    XT = np.transpose(X_tilde)
    theta_closed = np.dot(inv(np.dot(XT, X_tilde)), np.dot(XT, y))

    print("Q3:")
    print(theta_closed)

"""
Print the parameter vector θt once every 10 iterations during gradient descent.
Print your chosen learning rate and number of iterations.
Briefly describe your tuning process.
Questions 4(b), 4(c), 4(d), and 6(b) are open questions; you should determine and write reasonable answers based on your own analysis.
Use the following format:
    print("Q4a:")
    ...
    print("Q4b: " + str(your_learning_rate))
    print("Q4c: " + str(your_iterations))
    print("Q4d: ...")
In addition to the printed outputs above, save the loss plot to "loss_plot.jpg" using plt.savefig.
"""
def poly_regression_gradient():
    global losses
    n, d = X_tilde.shape
    theta = np.zeros(d)
    losses = []

    print("Q4a:")
    for t in range(iterations):

        if t % 10 == 0:
            print(theta)

        Y_hat = np.dot(X_tilde, theta)
        gradient = (1 / n) * np.dot(np.transpose(X_tilde), Y_hat - y)
        theta = theta - learning_rate * gradient

        loss = (1 / (2 * n)) * np.sum((np.dot(X_tilde, theta) - y) ** 2)
        losses.append(loss)

    chosen_learning_rate = learning_rate
    chosen_iterations = iterations

    print("Q4b: " + str(chosen_learning_rate))
    print("Q4c: " + str(chosen_iterations))
    closed_loss = (1/(2*n)) * np.sum((np.dot(X_tilde, theta_closed) - y)**2)
    print("Closed-form loss:", closed_loss)
    print("Final GD loss:", losses[-1])
    print("Difference:", abs(closed_loss - losses[-1]))

    tuning_desc = "Started with a smaller learning rate and increased it gradually while monitoring convergence behavior. The number of iterations was decreased until absolute value of loss difference was less than 10. The selected values converge smoothly without oscillation or divergence."
    print("Q4d: " + tuning_desc)

    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.jpg")
    plt.close()

"""
Print the model's prediction for the number of ice days for 2023-24 using the following format (include a space after ":"):
    print("Q5: " + str(y_hat))
"""
def prediction():
    x_2023 = 2023
    x_norm = (x_2023 - xmin) / (xmax - xmin)

    features = np.array([x_norm ** (i + 1) for i in range(degree)])
    features = np.append(features, 1)

    y_hat = np.dot(theta_closed, features)

    print("Q5: " + str(y_hat))

"""
Print the estimated local rate of change at year 2023 and your interpretation. Use the following format (include a space after ":"):
    print("Q6a: " + str(rate_of_change))
    print("Q6b: ...")
"""
def rate_of_change():
    x_norm = (2023 - xmin) / (xmax - xmin)

    rate = 0
    for i in range(degree):
        rate += (i + 1) * theta_closed[i] * (x_norm ** i)

    print("Q6a: " + str(rate))

    interpretation = "A positive value means the predicted number of ice days are increasing locally at year 2023. A negative value means the predicted number of ice days are decreasing locally at year 2023. A near zero value means the predicted number of ice days are relatively stable at year 2023."
    print("Q6b: " + interpretation)

if __name__ == "__main__":
    # Read in inputs
    filename = sys.argv[1]
    degree = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    iterations = int(sys.argv[4])

    data = pd.read_csv(filename)
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values

    data_visualization()
    data_normalization(degree)
    closed_poly_regression()
    poly_regression_gradient()
    prediction()
    rate_of_change()