import numpy as np
from root_numpy import root2array, array2root
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor



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



print 'creating NN'
regressor = MLPRegressor(
    hidden_layer_sizes=(100, ), 
    activation='relu', 
    solver='adam', 
    alpha=0.0001, 
    batch_size='auto', 
    learning_rate='constant', 
    learning_rate_init=0.001, 
    power_t=0.5, 
    max_iter=200, 
    shuffle=True, 
    random_state=1986, 
    tol=0.0001, 
    verbose=True, 
    warm_start=False, 
    momentum=0.9, 
    nesterovs_momentum=True, 
    early_stopping=True, 
    validation_fraction=0.1, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-08
)
print '\t==> created NN'


print 'start fitting'
regressor.fit(X_train.T, y_train)
print '\t==> fit done!'

print 'predicting...'
y_pred = regressor.predict(X_test.T)
print '\t==> The accuracy of this network is: %0.5f percent' % (abs(1. - y_pred/y_test)*100.).mean()


