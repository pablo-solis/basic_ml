import numpy as np
import util

from linear_model import LinearModel


def main(train_path, valid_path, save_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predictions using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)
    #check values...
    #Plot decision boundary on validation set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_valid, y_valid, clf.theta, save_path[:-4], correction=1)

    # Use np.savetxt to save predictions on eval set to save_path
    #need to add 1 intercept to x_train
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    np.savetxt(save_path, clf.predict(x_train))
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, clf.predict(x_valid))
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n, d).
            y: Training example labels. Shape (n,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        n = len(y)
        d = x.shape[1]
        #y = y.reshape([1,n])

        #phi  is scalar
        phi = np.mean(y)

        #mu_i.shape should = x[i].shape= (d,)
        mu_0 = (x.T@(1-y)) / (n*(1-phi))
        mu_1 = x.T@y / (n*phi) #shape (d,)

        '''
        phi = np.mean(y)
        self.cphi = phi
        mu_0 = np.transpose(x) @ (1 - y) / (n * (1 - phi))
        self.cmu_0 = mu_0
        mu_1 = np.transpose(x) @ y / (n * phi)
        self.cmu_1 = mu_1
        Sigma = np.zeros((d, d))
        for i in range(n):
            if y[i] == 1:
                cur_sample = (x[i] - mu_1).reshape(d, 1)
            else:
                cur_sample = (x[i] - mu_0).reshape(d, 1)
            Sigma = Sigma + 1 / n * (cur_sample @ np.transpose(cur_sample))
        self.cSigma = Sigma
        theta = (mu_1 - mu_0) @ np.linalg.pinv(Sigma)
        theta_0 = 1 / 2 * (np.transpose(mu_0) @ np.linalg.pinv(Sigma) @ mu_0 -
                           np.transpose(mu_1) @ np.linalg.pinv(Sigma) @ mu_1) - np.log((1 - phi) / phi)
        self.theta = np.concatenate((np.array([theta_0]), theta))'''


        #sigma.shape = (d,d)
        #sigma factors as (d,n)@(n,d)
        # (n,1)@(1,d) = (n,d)
        Mu = y.reshape([n,1])@mu_1.reshape([1,d]) + (1 - y).reshape([n,1]) @ mu_0.reshape([1,d])
        Xb = x- Mu #(n,d)
        sigma = (1/n)*Xb.T@Xb

        # Write theta in terms of the parameters
        # pinv = penrose inverse
        theta = np.linalg.pinv(sigma)@(mu_1 - mu_0) #(d,d)@(d,) = (d,)

        theta_0 = 1 / 2 * (mu_0 @ np.linalg.pinv(sigma) @ mu_0 - mu_1 @ np.linalg.pinv(sigma) @ mu_1) - np.log((1 - phi) / phi)
        self.theta = np.append([theta_0],theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n, d).

        Returns:
            Outputs of shape (n,).
        """
        # *** START CODE HERE ***
        n = x.shape[0]

        def sigmoid(x, theta):
            #x.shape = (1,d)
            #theta = (d,)
            z = x@theta
            return 1.0 / (1.0 + np.exp(-z))

        yvals = np.array([sigmoid(x[i], self.theta) for i in range(n)])
        return yvals
        # *** END CODE HERE

