import numpy as np

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict


class fit_methods:
    def __init__(self, X, y, fit_method="lasso", **kwargs):
        self.X = X
        self.y = y
        self.fit_method = fit_method
        self.frac = kwargs.get('frac', None)
        self.cv = kwargs.get('cv', None)

    def fit(self, **kwargs):
        if self.fit_method == "lasso":
            split_frac = kwargs.get('split_frac', 0.5)
            cv = kwargs.get('cv', 10)
            alpha, train_corr, test_corr, coef_, nnz_coef, mse = self.fit_lasso(split_frac=split_frac, cv=cv)
            return alpha, train_corr, test_corr, coef_, nnz_coef, mse
        

    def fit_lasso(self, split_frac = 0.5, cv = 10):
    
        # Split into training and test sets
        X_train, X_test , y_train, y_test = model_selection.train_test_split(self.X, 
                        self.y, test_size=split_frac, random_state=1)

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