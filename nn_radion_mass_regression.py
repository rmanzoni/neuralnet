import numpy as np
from lasagne import layers
from lasagne.updates import adam, adadelta
from lasagne.nonlinearities import linear
from nolearn.lasagne import NeuralNet    
from root_numpy import root2array, array2root
from root_numpy.testdata import get_filepath
from matplotlib import pyplot as plt
from nolearn.lasagne.visualize import plot_loss

signal_train = root2array('signals_training_enriched.root',start=0, stop=200000)
signal_test  = root2array('signals_test_enriched.root')
bkg_test     = root2array('ttbar_test_enriched.root')

np.random.shuffle(signal_train)
np.random.shuffle(signal_test )
np.random.shuffle(bkg_test    )

y_train = signal_train['gen_mass']
y_train = y_train.reshape(-1, 1).astype(np.float32)
X_train = np.array([
 signal_train['pt_mu'],
 signal_train['eta_mu'],
 signal_train['phi_mu'],
 signal_train['mass_mu'],
 signal_train['pt_tau'],
 signal_train['eta_tau'],
 signal_train['phi_tau'],
 signal_train['mass_tau'],
 signal_train['pt_bj1'],
 signal_train['eta_bj1'],
 signal_train['phi_bj1'],
 signal_train['mass_bj1'],
 signal_train['pt_bj2'],
 signal_train['eta_bj2'],
 signal_train['phi_bj2'],
 signal_train['mass_bj2'],
 signal_train['pt_jj'],
 signal_train['eta_jj'],
 signal_train['phi_jj'],
 signal_train['mass_jj'],
 signal_train['pt_sv'],
 signal_train['eta_sv'],
 signal_train['phi_sv'],
 signal_train['mass_sv'],
 signal_train['dr_bb'],
 signal_train['dr_bbsv'],
 signal_train['phi_met'],
 signal_train['pt_met'],
 ####################### 
 signal_train['mass_kf'],
 signal_train['chi2_kf'],
 signal_train['prob_kf'],
 signal_train['conv_kf'],
])




y_test = signal_test['gen_mass']
y_test = y_test.reshape(-1, 1).astype(np.float32)
X_test = np.array([
 signal_test['pt_mu'],
 signal_test['eta_mu'],
 signal_test['phi_mu'],
 signal_test['mass_mu'],
 signal_test['pt_tau'],
 signal_test['eta_tau'],
 signal_test['phi_tau'],
 signal_test['mass_tau'],
 signal_test['pt_bj1'],
 signal_test['eta_bj1'],
 signal_test['phi_bj1'],
 signal_test['mass_bj1'],
 signal_test['pt_bj2'],
 signal_test['eta_bj2'],
 signal_test['phi_bj2'],
 signal_test['mass_bj2'],
 signal_test['pt_jj'],
 signal_test['eta_jj'],
 signal_test['phi_jj'],
 signal_test['mass_jj'],
 signal_test['pt_sv'],
 signal_test['eta_sv'],
 signal_test['phi_sv'],
 signal_test['mass_sv'],
 signal_test['dr_bb'],
 signal_test['dr_bbsv'],
 signal_test['phi_met'],
 signal_test['pt_met'],
 #######################
 signal_test['mass_kf'],
 signal_test['chi2_kf'],
 signal_test['prob_kf'],
 signal_test['conv_kf'],
])



print 'input shape (samples, features)', X_train.T.shape
print 'target shape (samples,)'        , y_train.shape

print 'creating layers'

layers0 = [
    (layers.InputLayer  , { 'shape':(None, X_train.T.shape[1]) } ),
    (layers.DenseLayer  , { 'num_units':100, 'nonlinearity':linear }),
    (layers.DenseLayer  , { 'num_units':100, 'nonlinearity':linear }),
    (layers.DenseLayer  , { 'num_units':100, 'nonlinearity':linear }),
    (layers.DenseLayer  , { 'num_units':100, 'nonlinearity':linear }),
#     (layers.DenseLayer  , { 'num_units':100, 'nonlinearity':linear }),
    (layers.DenseLayer  , { 'num_units':1, 'nonlinearity':linear }), # output layer MUST have num_units=1
#     (layers.DropoutLayer, {'p':0.1 }),
#     (layers.DenseLayer  , { 'num_units':1, 'nonlinearity':linear }),
]

print '\t==> created layers'

print 'creating NN'
net = NeuralNet(
    layers=layers0, 
    regression=True, 
    update=adadelta, #adam,
    update_learning_rate=0.002,
    verbose=3,
    max_epochs=1000,
)
print '\t==> created NN'

print 'start fitting'
net.fit(X_train.T, y_train)
print '\t==> fit done!'

print 'predicting...'
y_pred = net.predict(X_test.T)
print '\t==> The accuracy of this network is: %0.5f percent' % (abs(1. - y_pred/y_test)*100.).mean()


print 'saving into augmented_signal_tree.root'
array2root(signal_test, 'augmented_signal_tree.root', 'tree', 'recreate')
fitted_mass = np.array(y_pred, dtype=[('fitted_mass', 'f4')])
array2root(fitted_mass, 'augmented_signal_tree_v2.root', 'tree', 'update')


