# AI534
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    data = pd.read_csv(path)
    return data


# Implements dataset preprocessing. For this assignment, you just need to implement normalization 
# of the three numerical features.
def preprocess_data(train, dev):
    # Your code here:
    column = ['Age', 'Annual_Premium', 'Vintage']
    column_mean = []
    column_std = []
    
    for col in column:
        column_mean.append(train[col].mean())
        column_std.append(train[col].std())
        train[col] = (train[col] - train[col].mean()) / train[col].std()
        

    for index, col in enumerate(column):
        dev[col] = (dev[col] - column_mean[index]) / column_std[index]
        
    return train, dev


# Implement other helper functions
def sigmoid_func(learned_hypo):
    return 1 / (1 + np.exp(-learned_hypo))

def cal_accuracy(y_pred, y_true):
    y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
    return (y_pred == y_true).mean()


# Trains a logistic regression model with L2 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L2_train(X, y, lr, lambd, dev_X, dev_y):
    # Your code here:
    weight = np.zeros(X.shape[1])
    stopping_threshold = 1e-8
    
    for itr in range(10000):
        gradient = ((X.multiply(y - sigmoid_func(X.dot(weight)), axis=0)).mean())
        weight = weight + lr * gradient
        weight[1:] = weight[1:] - lr * lambd * weight[1:]
        
        train_acc = cal_accuracy(sigmoid_func(X.dot(weight)), y)
        dev_acc = cal_accuracy(sigmoid_func(dev_X.dot(weight)), dev_y)
        
        loss = ((-1 * y * np.log(sigmoid_func(X.dot(weight)))) - ((np.ones(X.shape[0]) - y) * np.log(
            np.ones(X.shape[0]) - sigmoid_func(X.dot(weight))))).mean() + lambd * np.sum(np.power(weight[1:], 2))
        
        # print("iter={}, loss={}, train_acc={}, dev_acc={}".format(itr + 1, loss, train_acc, dev_acc))
        
        if np.linalg.norm(gradient) <= stopping_threshold:
            break
            
    train_acc = cal_accuracy(sigmoid_func(X.dot(weight)), y)
    dev_acc = cal_accuracy(sigmoid_func(dev_X.dot(weight)), dev_y)
    loss = ((-1 * y * np.log(sigmoid_func(X.dot(weight)))) - ((np.ones(X.shape[0]) - y) * np.log(
            np.ones(X.shape[0]) - sigmoid_func(X.dot(weight))))).mean() + lambd * np.sum(np.power(weight[1:], 2))
    
    print("iter={}, loss={}, regularizer={}, train_acc={}, dev_acc={}".format(itr + 1, loss, lambd, train_acc, dev_acc))
    
    return weight, train_acc, dev_acc



# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L1_train(X, y, lr, lambd, dev_X, dev_y):
    # Your code here:
    weight = np.zeros(X.shape[1])
    stopping_threshold = 1e-8
    
    for itr in range(10000):
        gradient = ((X.multiply(y - sigmoid_func(X.dot(weight)), axis=0)).mean())
        weight = weight + lr * gradient
        weight[1:] = np.sign(weight[1:]) * np.maximum(np.abs(weight[1:]) - (lr * lambd), np.zeros(weight[1:].shape))
        
        train_acc = cal_accuracy(sigmoid_func(X.dot(weight)), y)
        dev_acc = cal_accuracy(sigmoid_func(dev_X.dot(weight)), dev_y)
        
        loss = ((-1 * y * np.log(sigmoid_func(X.dot(weight)))) - ((np.ones(X.shape[0]) - y) * np.log(
            np.ones(X.shape[0]) - sigmoid_func(X.dot(weight))))).mean() + lambd * np.sum(np.abs(weight[1:]))
        
        # print("iter={}, loss={}, train_acc={}, dev_acc={}".format(itr + 1, loss, train_acc, dev_acc))
        
        if np.linalg.norm(gradient) <= stopping_threshold:
            break
            
    train_acc = cal_accuracy(sigmoid_func(X.dot(weight)), y)
    dev_acc = cal_accuracy(sigmoid_func(dev_X.dot(weight)), dev_y)
    loss = ((-1 * y * np.log(sigmoid_func(X.dot(weight)))) - ((np.ones(X.shape[0]) - y) * np.log(
    np.ones(X.shape[0]) - sigmoid_func(X.dot(weight))))).mean() + lambd * np.sum(np.power(weight[1:], 2))

    print("iter={}, loss={}, regularizer={}, train_acc={}, dev_acc={}".format(itr + 1, loss, lambd, train_acc, dev_acc))

    return weight, train_acc, dev_acc



# Generates and saves plots of the accuracy curves. Note that you can interpret accs as a matrix
# containing the accuracies of runs with different lambda values and then put multiple loss curves in a single plot.
def plot_losses(train_acc_df, dev_acc_df, flag):
    # Your code here:
    print('Regularization Plot...\t')

    fig, ax3 = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax3.semilogx(train_acc_df.regularizers, train_acc_df.accuracy,
                color='r', marker='o', markerfacecolor='m', zorder=1.5, alpha=0.5)

    ax3.semilogx(dev_acc_df.regularizers, dev_acc_df.accuracy,
                color='b', marker='x', markerfacecolor='r', zorder=1.5, alpha=0.5)

    min_axis = min(train_acc_df.accuracy.min(), dev_acc_df.accuracy.min())
    max_axis = max(train_acc_df.accuracy.max(), dev_acc_df.accuracy.max())

    ax3.set_ylabel(f'accuracy', color='r')
    ax3.set_xlabel(f'$\\lambda$, regularization size')
    ax3.set_xlim([1e-4, 1e4])
    ax3.set_ylim(0.495, 0.8)
    ax3.set_title(f"$\\mathcal{{L}}_2$, $\\alpha = {0.01}$, epochs = {10000}: Classification Accuracy",
                color='k', weight='normal', size=10)
    ax3.legend(["training", "validation"], loc="lower left")

    if flag == 1:
        plt.savefig("l2_train_dev_acc_cmp.jpg")
        print('Done.\n')

    if flag == 2:
        plt.savefig("l2_noisy_train_dev_acc_cmp.jpg")
        print('Done.\n')

    if flag == 3:
        plt.savefig("l1_train_dev_acc_cmp.jpg")
        print('Done.\n')
    return

# Invoke the above functions to implement the required functionality for each part of the assignment.
# Part 0  : Data preprocessing.
# Your code here:
# load in the training data
path1 = "IA2-train.csv"
train = load_data(path1)
# load in the validation data
path2 = "IA2-dev.csv"
dev = load_data(path2)
# load in the noisy training data
path3 = "IA2-train-noisy.csv"
noisy_data = load_data(path3)


# Part 1 . Implement logistic regression with L2 regularization and experiment with different lambdas
# Your code here:
""" Part (1a) """
weight_dict = defaultdict(list)
regularizers = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]

train, dev = preprocess_data(train, dev)

train_X = train.iloc[:, :-1]
train_y = train.iloc[:, -1]
dev_X = dev.iloc[:, :-1]
dev_y = dev.iloc[:, -1]
train_acc_array = []
dev_acc_array = []
    
for regularizer in regularizers:
    weight, train_acc, dev_acc = LR_L2_train(train_X, train_y, 0.01, regularizer, dev_X, dev_y)
    train_acc_array.append(round(train_acc, 3))
    dev_acc_array.append(round(dev_acc, 3))
    weight_dict[regularizer] = weight

train_acc_df = pd.DataFrame({'regularizers': regularizers, 'accuracy': train_acc_array})
dev_acc_df = pd.DataFrame({'regularizers': regularizers, 'accuracy': dev_acc_array})
l2 = True
plot_losses(train_acc_df, dev_acc_df, 1)


""" Part (1b) """
for lambd in [1e-3, 1e-2, 1e-1]:
    weight_ = weight_dict[lambd]
    sortedweight = weight_.abs().sort_values(ascending=False)[:5]
    biggest_weights = weight_.loc[list(sortedweight.index.values)]
    print("Top 5 features and their corresponding weights for lambda={}".format(lambd)) 
    print(biggest_weights, "\n")


""" Part (1c) """
nz_weight_count = []
regs = []
for key in weight_dict.keys():
    nz_weight_count.append(np.count_nonzero(weight_dict[key]==0) or np.count_nonzero(np.isnan(weight_dict[key])))
    regs.append(str(key))

fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
ax.bar(regs, nz_weight_count)
plt.title('Sparsity for different \u03BB')
plt.savefig('l2_sparsity.jpg')


# Part 2  Training and experimenting with IA2-train-noisy data.
# Your code here:
train_noisy, dev = preprocess_data(noisy_data, dev)
weight_dict = defaultdict(list)

train_X = train_noisy.iloc[:, :-1]
train_y = train_noisy.iloc[:, -1]
dev_X = dev.iloc[:, :-1]
dev_y = dev.iloc[:, -1]
train_acc_array = []
dev_acc_array = []
    
for regularizer in regularizers:
    weight, train_acc, dev_acc = LR_L2_train(train_X, train_y, 0.01, regularizer, dev_X, dev_y)
    train_acc_array.append(round(train_acc, 3))
    dev_acc_array.append(round(dev_acc, 3))
    weight_dict[regularizer] = weight

train_acc_df = pd.DataFrame({'regularizers': regularizers, 'accuracy': train_acc_array})
dev_acc_df = pd.DataFrame({'regularizers': regularizers, 'accuracy': dev_acc_array})
l2_noise = True
plot_losses(train_acc_df, dev_acc_df, 2)


# Part 3  Implement logistic regression with L1 regularization and experiment with different lambdas
# Your code here:
""" Part (3a) """
weight_dict = defaultdict(list)
train, dev = preprocess_data(train, dev)

train_X = train.iloc[:, :-1]
train_y = train.iloc[:, -1]
dev_X = dev.iloc[:, :-1]
dev_y = dev.iloc[:, -1]
train_acc_array = []
dev_acc_array = []
    
for regularizer in regularizers:
    weight, train_acc, dev_acc = LR_L1_train(train_X, train_y, 0.01, regularizer, dev_X, dev_y)
    train_acc_array.append(round(train_acc, 3))
    dev_acc_array.append(round(dev_acc, 3))
    weight_dict[regularizer] = weight

train_acc_df = pd.DataFrame({'regularizers': regularizers, 'accuracy': train_acc_array})
dev_acc_df = pd.DataFrame({'regularizers': regularizers, 'accuracy': dev_acc_array})
l1 = True
plot_losses(train_acc_df, dev_acc_df, 3)


""" Part (3a) """
for lambd in [1e-4, 1e-3, 1e-2]:
    weight_ = weight_dict[lambd]
    sortedweight = weight_.abs().sort_values(ascending=False)[:5]
    biggest_weights = weight_.loc[list(sortedweight.index.values)]
    print("Top 5 features and their corresponding weights for lambda={}".format(lambd)) 
    print(biggest_weights, "\n")

""" Part (3c) """
nz_weight_count = []
regs = []
for key in weight_dict.keys():
    nz_weight_count.append(np.count_nonzero(weight_dict[key]==0) or np.count_nonzero(np.isnan(weight_dict[key])))
    regs.append(str(key))

fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
ax.bar(regs, nz_weight_count)
plt.title('Sparsity for different \u03BB')
plt.savefig('l1_sparsity.jpg')