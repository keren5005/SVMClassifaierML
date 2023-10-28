import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from  SVM_Class import SVM
from Utiles_Functions import svm_primal, plot_classifier,svm_dual, train_test_split,polynomial_kernel,plot_classifier_z_kernel
from Utiles_Functions import highlight_support_vectors, plot_scores, gaussian_kernel, linear_kernel

" This function reads a dataset from a CSV file called 'simple_classification.csv', preprocesses the data, and fits an SVM using the primal " \
"form (svm_primal) to classify the data. It plots the decision boundary."
def q1a():
    df = pd.read_csv('simple_classification.csv')
    # preprocessing
    y = df['y'].apply(lambda x: -1 if x == 0 else 1).values

    x = df[['x1', 'x2']].values
    w = svm_primal(x, y)
    plt.title("Question 1a")
    plot_classifier(w, x, y)
    plt.show()

" this function reads the same dataset and fits an SVM using the " \
"dual form (svm_dual). It plots the decision boundary."
def q1b():
    df = pd.read_csv('simple_classification.csv')
    # preprocessing
    y = df['y'].apply(lambda x: -1 if x == 0 else 1).values
    x = df[['x1', 'x2']].values

    alpha, w = svm_dual(x, y, return_w=True)
    plt.title("Question 1b")
    plot_classifier(w, x, y)
    plt.show()

"A helper function used in q2(). It splits the data into training and testing sets, fits an SVM with a specified kernel, computes and prints the " \
"classification score, and plots the decision boundary with support vectors."
def q2_helper(x, y, title, kernel):
    test_x, train_x, test_y, train_y = train_test_split(0.2, x, y, random_state=42)
    # here we are using the SVM class from the question 3
    svm = SVM(kernel=kernel)
    svm.fit(train_x, train_y)
    score = svm.score(test_x, test_y)
    print(f'{title} score = {score}')
    alpha = svm.alphas
    # alpha = svm_dual_kernel(x, y, kernel)
    plt.title(title)
    plot_classifier_z_kernel(alpha, x, y, polynomial_kernel, s=80)
    highlight_support_vectors(x, alpha)
    plt.show()
    return score

"This function reads a dataset from a CSV file called 'simple_nonlin_classification.csv' " \
"and performs SVM classification using different kernels. "
def q2():
    df = pd.read_csv('simple_nonlin_classification.csv')
    # preprocessing
    y = df['y'].values
    x = df[['x1', 'x2']].values

    data = {}

    data['P(3,5)'] = q2_helper(x, y, "Q2 - Polynomial kernel c=3, d=5", lambda x1, x2: polynomial_kernel(x1, x2, 3, 5))
    data['P(1,3)'] = q2_helper(x, y, "Q2 - Polynomial kernel c=1, d=3", lambda x1, x2: polynomial_kernel(x1, x2, 1, 3))
    data['P(3,3)'] = q2_helper(x, y, "Q2 - Polynomial kernel c=3, d=3", lambda x1, x2: polynomial_kernel(x1, x2, 3, 3))
    data['G(.5)'] = q2_helper(x, y, "Q2 - RBF kernel gamma=0.5", lambda x1, x2: gaussian_kernel(x1, x2, 0.5))
    data['G(4)'] = q2_helper(x, y, "Q2 - RBF kernel gamma=4", lambda x1, x2: gaussian_kernel(x1, x2, 4))

    plot_scores(data.keys(), data.values(), 'simple_nonlin_classification.csv')

"This function reads a dataset from a CSV file called 'wisconsin.csv', preprocesses the data by scaling it using StandardScaler, and performs SVM classification using different kernels (linear, polynomial, and Gaussian) with varying parameter values. " \
"It prints the classification scores and plots the results."
def q4():
    df = pd.read_csv("wisconsin.csv")
    y = df['diagnosis'].apply(lambda x: -1 if x == 0 else 1).values
    x = df.drop(['diagnosis'], axis=1).values
    # normalize values to improve numeric stability errors
    x_scaled = StandardScaler().fit_transform(x)
    test_x, train_x, test_y, train_y = train_test_split(0.2, x_scaled, y, random_state=42)
    kernels = {
        "linear kernel": (lambda x1, x2: linear_kernel(x1, x2), 'Linear'),
        "polynomial kernel (C = 1, d = 2)": (lambda x1, x2: polynomial_kernel(x1, x2, 1, 2), 'P(1,2)'),
        "polynomial kernel (C = 1, d = 3)": (lambda x1, x2: polynomial_kernel(x1, x2, 1, 3), 'P(1, 3)'),
        "polynomial kernel (C = 1, d = 4)": (lambda x1, x2: polynomial_kernel(x1, x2, 1, 4), 'P(1, 4)'),
        "gaussian kernel (gamma = 0.1)": (lambda x1, x2: gaussian_kernel(x1, x2, 0.1), 'G(.1)'),
        "gaussian kernel (gamma = 0.5)": (lambda x1, x2: gaussian_kernel(x1, x2, 0.5), 'G(.5)'),
        "gaussian kernel (gamma = 1)": (lambda x1, x2: gaussian_kernel(x1, x2, 1), 'G(1)'),
        "gaussian kernel (gamma = 4)": (lambda x1, x2: gaussian_kernel(x1, x2, 4), 'G(4)')
    }
    names = []
    scores = []
    for k in kernels.keys():
        kern, name = kernels[k]
        svm = SVM(kernel=kern)
        svm.fit(train_x, train_y)
        score = svm.score(test_x, test_y)
        names.append(name)
        scores.append(score)
        print(f'Q4 - {k} score = {score}')

    plot_scores(names, scores, "Wisconsin")


if __name__ == "__main__":
    # silence UserWarning: Converted P to scipy.sparse.csc.csc_matrix
    # For best performance, build P as a scipy.sparse.csc_matrix rather than as a numpy.ndarray
    warnings.filterwarnings("ignore")

    q1a()
    q1b()
    q2()
    print("===================================================================")
    q4()
