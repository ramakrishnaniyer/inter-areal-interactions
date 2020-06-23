import numpy as np

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

def load_stimulus_filtered_array(stim_arr_fname, stim_durn, dt):
    upsmp_fac = round(stim_durn/dt) 
    print(stim_durn, dt, upsmp_fac)
    stim_filt_arr = np.load(stim_arr_fname)
    print(stim_filt_arr.shape)

    stim_filt_arr_upsmp = np.repeat(stim_filt_arr.T, upsmp_fac, axis=1)

    return stim_filt_arr_upsmp

def fit_lasso(X, y, split_frac = 0.5, cv = 10):
    
    # Split into training and test sets
    X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=split_frac, random_state=1)

    #Lasso CV
    lasso = Lasso(max_iter = 10000, normalize = False, fit_intercept=False)
    lassocv = LassoCV(alphas = None, cv = cv, max_iter = 10000, normalize = False, fit_intercept=False)
    lassocv.fit((X_train), y_train)

    lasso.set_params(alpha=lassocv.alpha_)
    lasso.fit((X_train), y_train)
    pred_train = np.squeeze(lasso.predict((X_train)))
    pred = np.squeeze(lasso.predict((X_test)))

    train_corr = np.corrcoef(y_train,pred_train)[0,1]
    mse = mean_squared_error(y_test, pred)
    test_corr = np.corrcoef(y_test,pred)[0,1]
    nnz_coef = len(lasso.coef_[np.abs(lasso.coef_>1e-10)])

    return lassocv.alpha_, train_corr, test_corr, lasso.coef_, nnz_coef, mse