import numpy as np
import util

from linear_model import LinearModel


def main(train_path, valid_path, save_path):
    """Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predictions using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    clf = LogisticRegression()
    theta = clf.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    #load the validation data
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    # apply trained model to validation set
    #remove '.txt' from save_path
    util.plot(x_valid, y_valid, theta, save_path[:-4], correction=1)

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path,clf.predict(x_valid))


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """


    #used in self.fit
    def one_norm(self, vec):
        return sum([abs(v) for v in vec])

    #used in self.predict
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    #derivative of sigmoid function, used in self.Hess_J
    def der_sig(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    #used in self.predict
    def model(self, theta, x):
        # x is an np.array shape (1,d)
        #theta has shape (d,1)
        # theta is an np.array of the same length
        return self.sigmoid(theta.T @ x)

    #shape (d,1)
    def grad_J(self,theta,x,y):
        # return the gradient of J evaluated at theta
        #for gradient need vec with entries
        #(y[i] - self.model(theta,x[i])
        vec = np.array([yx[0] - self.model(theta, yx[1]) for yx in zip(y, x)])

        # x.T has shape (d,n)
        # vec has shape(n,1)
        # output has shape (d,1)
        n = len(y)
        return (-1 / n) * (x.T @ vec)

    #shape(d,d)
    def Hess_J(self,theta,x,y):
        # return the Hessian of J evaluated at theta
        # x[i]@theta has shape (1,1) use as scalar to return value
        n = len(y)
        lst= [self.der_sig(np.asscalar(x[i]@theta)) for i in range(n)]
        D = np.diag(lst)
        # x is nxd and x.T is dxn
        return (1 / n) * (x.T @ D @ x)



    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n, d).
            y: Training example labels. Shape (n,).
        Returns:
            output of shape (d,1)
        """
        # set intitial theta as col vec
        d = x.shape[1]
        n = len(y)
        theta = np.zeros([d,1])

        Hinv = np.linalg.inv(self.Hess_J(theta, x, y)) #shape (d,d)

        update = -Hinv @ self.grad_J(theta, x, y) #shape (d,1)
        theta += update
        test = self.one_norm(update)

        while (test > 1e-5):
            # use update rule:
            Hinv = np.linalg.inv(self.Hess_J(theta, x, y))
            update = -Hinv @ self.grad_J(theta, x, y)
            theta += update
            test = self.one_norm(update)
        #shape (d,1)
        self.theta = theta
        return theta

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n, d).

        Returns:
            Outputs of shape (n,).
        """
        yvals = np.array([self.model(self.theta, xi) for xi in x])
        return yvals
