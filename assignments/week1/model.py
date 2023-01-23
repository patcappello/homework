import numpy as np


class LinearRegression:
    """
    A linear regression model that uses the linear regression closed form to fit the model.
    """
    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = 0.00

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the given input.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input labels.

        Returns:
            np.ndarray: The weight vector.
            np.ndarray: The bias value.

        """
        m, n = X.shape
        X = np.concatenate((np.ones((m,1)), X), axis=1)
        soln = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.w, self.b = soln[1:], soln[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def _standardize(self, X, y):
        return (X-np.mean(X))/np.std(X), (y-np.mean(y))/np.std(y)

    def _calcGrads(self, X, y):
        m, n = X.shape
        # m is the number of examples, n is the number of features
        y_pred = self.predict(X)
        dw = (2/m)*(X.T @ (y_pred-y))
        db = (2/m)*np.sum(y_pred-y)
        return dw, db

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the model to the given input.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input labels.
            lr (float): The learning rate
            epochs (int): The number of training epochs

        Returns:
            np.ndarray: The weight vector.
            np.ndarray: The bias value.

        """
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        decay_rate = 1e-3
        X, y = self._standardize(X, y)
        for i in range(epochs):
            lr = lr/(1+decay_rate*i)
            dw, db = self._calcGrads(X, y)
            self.w -= lr*dw
            self.b -= lr*db
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b