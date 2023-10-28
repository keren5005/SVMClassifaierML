import itertools
import random
import qpsolvers as qps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_data(X, y, zoom_out=False, s=None):
    if zoom_out:
        x_min = np.amin(X[:, 0])
        x_max = np.amax(X[:, 0])
        y_min = np.amin(X[:, 1])
        y_max = np.amax(X[:, 1])
        plt.axis([x_min - 1, x_max + 1, y_min - 1, y_max + 1])
    plt.scatter(X[:, 0], X[:, 1], c=y, s=s, cmap=matplotlib.colors.ListedColormap(['blue', 'red']))

"Plots the decision boundary of a classifier"
def plot_classifier(w, X, y):
    plot_data(X, y)

    lx = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 60)

    ly = [(-w[-1] - w[0] * p) / w[1] for p in lx]
    plt.plot(lx, ly, color='black')

    ly1 = [(-w[-1] - w[0] * p - 1) / w[1] for p in lx]
    plt.plot(lx, ly1, "--", color='red')

    ly2 = [(-w[-1] - w[0] * p + 1) / w[1] for p in lx]
    plt.plot(lx, ly2, "--", color='blue')


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2, C=1, d=3):
    return (np.dot(x1, x2) + C) ** d

"Computes the Gaussian (RBF) kernel between two input vectors."
def gaussian_kernel(x, y, gamma=0.5):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

"Implements SVM using the dual formulation and solves it using quadratic programming."
def svm_dual(X, y, max_iter=4000, verbose=False, return_w=False):
    N = X.shape[0]
    X = np.c_[X, np.ones(N)]
    G = np.diag(y) @ X
    P = 0.5 * G @ G.T
    q = -np.ones(N)
    GG = -np.eye(N)
    h = np.zeros(N)

    alpha = qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    # w = \sum_i alpha_iy_ix_i
    w = G.T @ alpha
    if not return_w:
        return alpha  # , 0.5 * w
    return alpha, 0.5 * w

"Implements SVM using the dual formulation with a kernel function."
def svm_dual_kernel(X, y, ker, max_iter=4000, verbose=False):
    N = X.shape[0]
    P = np.empty((N, N))
    for i, j in itertools.product(range(N), range(N)):
        P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :])
    P = 0.5 * (P + P.T)
    P = 0.5 * P
    q = -np.ones(N)
    GG = -np.eye(N)
    h = np.zeros(N)

    alpha = qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    # w = \sum_i alpha_iy_ix_i
    # w = P.T @ alpha

    return alpha  # , 0.5 * w


def soft_svm_dual(X, y, C=1.0, max_iter=4000, verbose=False):
    N = X.shape[0]
    X = np.c_[X, np.ones(N)]
    G = np.diag(y) @ X
    P = 0.5 * G @ G.T
    q = -np.ones(N)
    GG = np.block([[-np.eye(N)], [np.eye(N)]])
    h = np.block([np.zeros(N), C * np.ones(N)])

    alpha = qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    # w = \sum_i alpha_iy_ix_i
    w = G.T @ alpha

    return alpha  # , 0.5 * w


def soft_svm_dual_kernel(X, y, ker, C=1.0, max_iter=4000, verbose=False):
    N = X.shape[0]
    X = np.c_[X, np.ones(N)]
    G = np.diag(y) @ X
    P = np.empty((N, N))
    for i, j in itertools.product(range(N), range(N)):
        P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :])
    P = 0.5 * (P + P.T)

    q = -np.ones(N)
    GG = np.block([[-np.eye(N)], [np.eye(N)]])
    h = np.block([np.zeros(N), C * np.ones(N)])

    alpha = qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    # w = \sum_i alpha_iy_ix_i
    # w = G.T @ alpha

    return alpha

"Implements SVM using the primal formulation and solves it using quadratic programming."
def svm_primal(X, y, max_iter=4000, verbose=False):
    N = X.shape[0]
    n = 1 + X.shape[1]
    X = np.c_[X, np.ones(N)]
    P = np.eye(n)
    q = np.zeros(n)
    G = -np.diag(y) @ X
    h = -np.ones(N)
    w = qps.solve_qp(P, q, G, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    return w

"Plots the decision boundary of a classifier when using a kernel function."
def plot_classifier_z_kernel(alpha, X, y, ker, s=None):
    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])
    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])

    xx = np.linspace(x_min, x_max)
    yy = np.linspace(y_min, y_max)

    xx, yy = np.meshgrid(xx, yy)

    # N = X.shape[0]
    N = len(alpha)
    z = np.zeros(xx.shape)
    for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
        v = sum([y[k] * alpha[k] * ker(X[k, :], np.array([xx[i, j], yy[i, j]])) for k in range(N)])
        z[i, j] = v

    plt.rcParams["figure.figsize"] = [15, 10]

    plt.contour(xx, yy, z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], linestyles=['--', '-', '--'])

    plot_data(X, y, s=s)

" Returns the indices of the support vectors."
def support_vectors(alpha, thresh=0.0001):
    return np.argwhere(np.abs(alpha) > thresh).reshape(-1)

"Highlights the support vectors in a plot"
def highlight_support_vectors(X, alpha):
    sv = support_vectors(alpha)
    plt.scatter(X[sv, 0], X[sv, 1], s=300, linewidth=1, facecolors='none', edgecolors='g')

# ===============================================================================
" Splits the data into training and test sets."
def train_test_split(test_size: float, X, y, random_state=None):
    n = len(y)
    indices = [i for i in range(n)]
    if random_state is not None:
        random.seed(random_state)
    random.shuffle(indices)
    test_idx = int(n * test_size)
    return X[indices[0:test_idx]], X[indices[test_idx:-1]], y[indices[0:test_idx]], y[indices[test_idx:-1]]


# ===============================================================================
"Plots the scores obtained from different SVM models."
def plot_scores(names, scores, title):
    plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(names, scores, color='maroon', width=0.4)

    plt.xlabel("Kernel")
    plt.ylabel("Score")
    plt.title(f"Scores for {title} dataset")
    plt.show()
