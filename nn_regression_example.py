import numpy as np

from matplotlib import pyplot as plt

from lasagne import layers
from lasagne.updates import adam, adadelta

from nolearn.lasagne import NeuralNet    
from nolearn.lasagne.visualize import plot_loss
from sklearn.datasets import make_regression
from sklearn.cross_validation import train_test_split

# create a training dataset
X, y, coef = make_regression(n_samples=1000, n_features=30, noise=0.15, coef=True, random_state=1986)
y = y.reshape(-1, 1).astype(np.float32)

# split test/train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1986)

print 'input shape (samples, features)', X_train.shape
print 'target shape (samples,)'        , y_train.shape

layers0 = [
    (layers.InputLayer  , { 'name':'input'  , 'shape':(None, X_train.shape[1])}),
    (layers.DenseLayer  , { 'name':'hidden' , 'num_units':1000, 'nonlinearity':None }),
    (layers.DenseLayer  , { 'name':'output' , 'num_units':1, 'nonlinearity':None }), # being a single-target regression, the output layer MUST have num_units=1
]

net = NeuralNet(
    layers=layers0, 
    regression=True, 
    update=adam,
    update_learning_rate=0.005,
    verbose=3,
    max_epochs=10,
)

net.fit(X_train, y_train)

y_pred = net.predict(X_test)
print "The accuracy of this network is: %0.5f percent" % (abs(1. - y_pred/y_test)*100.).mean()


# plot loss function value as a function of epoch for the training and test samples
# plot_loss(net)

# scatter plot between true and predicted values
plt.scatter(y_pred, y_test)

plt.show()


