from matplotlib.backends import backend_pdf
import csv
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio

# I/O Constants
TRAINING_DATA_FILENAME = "joke_data/joke_train.mat"
VALIDATION_DATA_FILENAME = "joke_data/validation.txt"
TEST_DATA_FILENAME = "joke_data/query.txt"

U_FILENAME = "svdmat/U_"
S_FILENAME = "svdmat/S_"
V_FILENAME = "svdmat/V_"
X_HAT_FILENAME = "svdmat/X_"
BASIC_TAG = "_basic.csv"
ADVANCED_TAG = "_advanced.csv"

# Hyperparameters
LAMBDA = 3200
D_VALUES = [2, 5, 10, 20]
EPSILON_U = 1e-5
EPSILON_X_HAT = 1e-5

# Data Processing
def load_training_data():
    return scipy.io.loadmat(TRAINING_DATA_FILENAME)['train']

def load_validation_data():
    with open(VALIDATION_DATA_FILENAME) as f:
        user_joke_pairs, labels_validation = [], []
        for row in f:
            i, j, label = [int(x) for x in row.split(",")]
            user_joke_pairs.append((i, j))
            labels_validation.append(label)
    return user_joke_pairs, np.array(labels_validation)

def load_test_data():
    with open(TEST_DATA_FILENAME) as f:
        ids, user_joke_pairs = [], []
        for row in f:
            id, i, j = [int(x) for x in row.split(",")]
            ids.append(id)
            user_joke_pairs.append((i, j))
    return ids, user_joke_pairs

def save_SVD(U, S, V, d, tag):
    with open(U_FILENAME + str(d) + tag, 'w') as f:
        w1 = csv.writer(f, delimiter=',', quotechar='|')
        for i in range(U.shape[0]):
            w1.writerow([U[i][j] for j in range(U.shape[1])])

    with open(S_FILENAME + str(d) + tag, 'w') as f:
        w2 = csv.writer(f, delimiter=',', quotechar='|')
        w2.writerow([S[i] for i in range(S.shape[0])])

    with open(V_FILENAME + str(d) + tag, 'w') as f:
        w3 = csv.writer(f, delimiter=',', quotechar='|')
        for i in range(V.shape[0]):
            w3.writerow([V[i][j] for j in range(V.shape[1])])

def save_SVD_2(U, X_hat, d, tag):
    with open(U_FILENAME + str(d) + tag, 'w') as f:
        w1 = csv.writer(f, delimiter=',', quotechar='|')
        for i in range(U.shape[0]):
            w1.writerow([U[i][j] for j in range(U.shape[1])])

    with open(X_HAT_FILENAME + str(d) + tag, 'w') as f:
        w2 = csv.writer(f, delimiter=',', quotechar='|')
        w2.writerow([X_hat[i] for i in range(X_hat.shape[0])])

def load_SVD(d, tag):
    with open(U_FILENAME + str(d) + tag, 'r') as f:
        r1 = csv.reader(f, delimiter=',', quotechar='|')
        U = np.array([np.array([float(x) for x in row]) for row in r1])

    with open(S_FILENAME + str(d) + tag, 'r') as f:
        r2 = csv.reader(f, delimiter=',', quotechar='|')
        S = np.array([[float(x) for x in row] for row in r2][0])

    with open(V_FILENAME + str(d) + tag, 'r') as f:
        r3 = csv.reader(f, delimiter=',', quotechar='|')
        V = np.array([np.array([float(x) for x in row]) for row in r3])

    return U, S, V

def load_SVD_2(d, tag):
    with open(U_FILENAME + str(d) + tag, 'r') as f:
        r1 = csv.reader(f, delimiter=',', quotechar='|')
        U = np.array([np.array([float(x) for x in row]) for row in r1])

    with open(X_HAT_FILENAME + str(d) + tag, 'r') as f:
        r3 = csv.reader(g, delimiter=',', quotechar='|')
        X_hat = np.array([np.array([float(x) for x in row]) for row in r3])

    return U, X_hat
   
def preprocess(data):
    return np.nan_to_num(data)

# Training functions
def learn_basic(R, d, tag):
    U, S, V = scipy.sparse.linalg.svds(R, k=d)
    save_SVD(U, S, V, d, tag)
    return U, S, np.transpose(V)

def learn_advanced(U, S, V, R, d, tag):
    s, X_hat, i = np.isnan(R), V.T, 0
    while True:
        print("Iteration %d" % i)
        print("Updating U.")
        U, prev_U = update_U(U.copy(), X_hat, s, R), U
        print("Updating X_hat.")
        X_hat, prev_X_hat = update_X_hat(U, X_hat.copy(), s, R), X_hat

        print(np.sum(np.square(U - prev_U)))
        print("%f\n" % np.sum(np.square(X_hat - prev_X_hat)))
        if is_converged(U, prev_U, EPSILON_U) and is_converged(X_hat, prev_X_hat, EPSILON_X_HAT):
            break
        i += 1
    save_SVD_2(U, X_hat, d, tag)
    return U, X_hat

def update_U(U, X_hat, S, R):
    X_hat = X_hat.T
    for i in range(U.shape[0]):
        outer_prod_sum, R_sum = np.zeros((X_hat.shape[0], X_hat.shape[0])), np.zeros(X_hat.shape[0]).T
        for j in range(S.shape[1]):
            if not S[i][j]:
                outer_prod_sum += np.dot(X_hat.T[j], X_hat.T[j].T)
                R_sum += R[i][j]*X_hat.T[j]
        U[i] = scipy.linalg.solve(outer_prod_sum + LAMBDA*np.identity(outer_prod_sum.shape[0]), R_sum)
    return U 

def update_X_hat(U, X_hat, S, R):
    for j in range(X_hat.shape[1]):
        outer_prod_sum, R_sum = np.zeros((U.shape[1], U.shape[1])), np.zeros(U.T.shape[0]).T
        for i in range(S.shape[0]):
            if not S[i][j]:
                outer_prod_sum += np.dot(U[i], U[i].T)
                R_sum += R[i][j]*U[i]
        X_hat[j] = scipy.linalg.solve(outer_prod_sum + LAMBDA*np.identity(outer_prod_sum.shape[0]), R_sum)
    return X_hat

def is_converged(A, prev_A, epsilon):
    return np.sum(np.square(A - prev_A)) < epsilon

def predict(user_joke_pairs, U, X_hat):
    return np.array([max(np.sign(np.dot(U[i-1], X_hat[j-1])), 0) for i, j in user_joke_pairs])

def compute_mse(R, U, S, V):
    s, mse, X_hat = np.isnan(R), 0, np.dot(V, np.diag(S))
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if not s[i][j]:
                mse += (np.dot(U[i], X_hat[j]) - R[i][j])**2
    return mse

if __name__ == '__main__':
    # Latent Factor Model with basic preprocessing
    R = preprocess(load_training_data())

    user_joke_pairs_validation, labels_validation = load_validation_data()
    mse_values, validation_accuracies = [], []
    for d in D_VALUES:
        U, S, V = learn_basic(R, d, BASIC_TAG)
        mse = compute_mse(load_training_data(), U, S, V)
        mse_values.append(mse)
        print("MSE for d=%d: %f" % (d, mse))

        predictions_validation = predict(user_joke_pairs_validation, U, np.dot(V, np.diag(S)))
        validation_accuracy = metrics.accuracy_score(labels_validation, predictions_validation)
        validation_accuracies.append(validation_accuracy)
        print("Validation accuracy for d={0}: {1}".format(d, validation_accuracy))
    plt.figure()
    plt.plot(D_VALUES, mse_values)
    plt.title("MSE vs. d")
    plt.ylabel("MSE")
    plt.xlabel("d")
    plt.xlim((1, 21))
    with backend_pdf.PdfPages("mse_plot.pdf") as pdf:
        pdf.savefig()

    plt.figure()
    plt.plot(D_VALUES, validation_accuracies)
    plt.title("Validation Accuracy vs. d")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("d")
    plt.xlim((1, 21))
    with backend_pdf.PdfPages("accuracies_plot.pdf") as pdf:
        pdf.savefig()

    # Latent Factor Model with advanced preprocessing
    for d in D_VALUES:
        U, S, V = load_SVD(d, BASIC_TAG)
        U, X_hat = learn_advanced(U, S, V, load_training_data(), d, ADVANCED_TAG)
        
        mse = compute_mse(load_training_data(), U, S, X_hat)
        print("MSE for d=%d: %f" % (d, mse))
        
        predictions_validation = predict(user_joke_pairs_validation, U, X_hat)
        print("Validation accuracy for d={0}: {1}".format(d, metrics.accuracy_score(labels_validation, predictions_validation)))

    