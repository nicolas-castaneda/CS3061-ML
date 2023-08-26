# Import libraries
import pandas
import numpy as np

import matplotlib.pyplot as plt

class Regression:
    # Mean Squared Error
    # n = number of elements
    # sum( (Y - Y_a)**2 ) / n
    def MSE(self, X, Y, B):
        Y_approximation = X.dot(B)
        rows, columns = X.shape
        return (np.sum((Y - Y_approximation)**2))/rows
        
    # Adds a column of ones on the leftmost part of a matrix 
    def add_bias(self, X):
        rows, columns = X.shape
        X0 = np.ones((rows, 1))
        return np.hstack((X0, X))

    # Beta = ( ( X_T . X )^-1 ) . X_t . Y 
    def calculate_beta(self, X, Y):
        X_t = np.transpose(X)
        return np.linalg.inv(np.dot(X_t, X)).dot(X_t).dot(Y)

class Graphic:
    # Function
    # Y = b0 + b1 * X1 + b2 * X2
    def f(x1, x2, Beta):
        x0 = 0
        if ( Beta.shape[0] == 3):
            x0 = Beta[-3]
        return x0 + x1 * Beta[-2] + x2 * Beta[-1]

    def graph(self, X, Y, Beta):
        # Create the figure
        fig = plt.figure()

        # Add axes
        ax = fig.add_subplot(111, projection='3d')

        # Set labels
        ax.set_xlabel("Year")
        ax.set_ylabel("Kms_Driven")
        ax.set_zlabel("Present_Price")

        # Plot points
        ax.scatter(X[:,-2], X[:,-1], Y)
        plt.xticks(np.arange(X[:,-2].min(), X[:,-2].max(), 2.0))

        # Set plane boundaries
        x1 = np.linspace(X[:,-2].min(), X[:,-2].max(), 100)
        x2 = np.linspace(X[:,-1].min(), X[:,-1].max(), 100)
        x1, x2 = np.meshgrid(x1, x2)
        z = Graphic.f(x1, x2, Beta)

        # Plot surface
        ax.plot_surface(x1, x2, z, alpha=0.6, edgecolor='none')

        plt.show()


# Create objects 
graphic = Graphic()
regression = Regression()

# Load data
dataframe = pandas.read_csv("car_data.csv")
X = (dataframe[["Year", "Kms_Driven"]]).to_numpy()
Y = (dataframe['Present_Price']).to_numpy()

# Unbiased Regression
unbiased_beta = regression.calculate_beta(X, Y)
unbiased_MSE = regression.MSE(X, Y, unbiased_beta)
print("Unbiased Beta:", unbiased_beta)
print("Unbiased MSE:", unbiased_MSE)
graphic.graph(X, Y, unbiased_beta)

# Biased Regression
X = regression.add_bias(X)

biased_beta = regression.calculate_beta(X, Y)
biased_MSE = regression.MSE(X, Y, biased_beta)
print("Biased Beta:", biased_beta)
print("Biased MSE:", biased_MSE)
graphic.graph(X, Y, biased_beta)
