import numpy as np
from Utiles_Functions import linear_kernel, polynomial_kernel, gaussian_kernel, svm_dual,svm_dual_kernel,soft_svm_dual, soft_svm_dual_kernel

class SVM:
    def __init__(self, C=None, kernel=None, degree=None, gamma=None):
        "The constructor method that initializes the SVM object with the specified parameters."
        self.C = C
        if kernel is None:
            self.kernel = None
        else:
            if isinstance(kernel, str):
                if kernel == 'linear':
                    self.kernel = linear_kernel
                elif kernel == 'poly':
                    self.kernel = lambda x1, x2: polynomial_kernel(x1, x2, 0, degree)
                elif kernel == 'rbf':
                    self.kernel = lambda x1, x2: gaussian_kernel(x1, x2, gamma)
                else:
                    raise Exception(f'Unknown kernel {kernel}')
            else:
                self.kernel = kernel

        self.alphas = None
        self.b = 0
        self.support_x = None
        self.support_y = None

    def fit(self, x, y):
        "Trains the SVM classifier on the input data x and corresponding labels y."
        if self.C is None:
            if self.kernel is None:
                lagrange_multipliers = svm_dual(x, y)
            else:
                lagrange_multipliers = svm_dual_kernel(x, y, self.kernel)
        else:
            if self.kernel is None:
                lagrange_multipliers = soft_svm_dual(x, y, self.C)
            else:
                lagrange_multipliers = soft_svm_dual_kernel(x, y, self.kernel, self.C)

        sv = lagrange_multipliers > 1e-5
        ind = np.arange(len(lagrange_multipliers))[sv]
        self.alphas = lagrange_multipliers[sv]
        self.support_y = y[sv]
        self.support_x = x[sv]

        n_samples, n_features = x.shape
        K = np.zeros((n_samples, n_samples))
        kern = self.get_kernel()
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = kern(x[i], x[j])

        self.b = 0
        for n in range(len(self.alphas)):
            # For all support vectors:
            self.b += self.support_y[n]
            self.b -= np.sum(self.alphas * self.support_y * K[ind[n], sv])
        self.b = self.b / len(self.alphas)

    def __distance_from_hyperplane(self, X):
        "Computes the distance of the input samples X from the SVM hyperplane."
        y_predict = np.zeros(len(X))
        kern = self.get_kernel()
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alphas, self.support_y, self.support_x):
                s += a * sv_y * kern(X[i], sv)
            y_predict[i] = s
        return y_predict + self.b

    def predict(self, X):
        " Predicts the labels for the input samples X."
        return np.sign(self.__distance_from_hyperplane(X))

    def decision_function(self, x):
        "Computes the decision function values for the input samples x."
        dist = self.__distance_from_hyperplane(x)
        return np.c_[x, dist, np.sign(dist)]

    def score(self, X, y):
        "Calculates the classification accuracy of the SVM on the input data X and corresponding labels y"
        y_pred = self.predict(X)
        errors = 0
        for i in range(len(y)):
            if y_pred[i] != y[i]:
                errors += 1
        return 1 - (errors / len(y))

    def get_kernel(self):
        "Returns the kernel function used by the SVM."
        if self.kernel is None:
            return linear_kernel
        return self.kernel