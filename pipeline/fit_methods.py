import numpy as np

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

from scipy import around
from scipy import size
from scipy.linalg import norm



class fit_methods:
    def __init__(self, X, y, fit_method="lasso", **kwargs):
        self.X = X
        self.y = y
        self.fit_method = fit_method
        self.frac = kwargs.get('frac', None)
        self.cv = kwargs.get('cv', None)

    def fit(self, **kwargs):
        
        split_frac = kwargs.get('split_frac', 0.5)
        cv = kwargs.get('cv', 10)
        
        if self.fit_method == "lasso":  
            alpha, train_corr, test_corr, coef_, nnz_coef, mse = self.fit_lasso(split_frac=split_frac, cv=cv)
            return alpha, train_corr, test_corr, coef_, nnz_coef, mse
        elif self.fit_method == "ols":
            pred = self.fit_ols(split_frac=split_frac)
        elif self.fit_method == 'pcr':
            pred = self.fit_pcr(split_frac=split_frac, cv=cv)
        elif self.fit_method == 'pls':
            pred = self.fit_pls(split_frac=split_frac, cv=cv)
        elif self.fit_method == 'rrr':
            pred = self.fit_rrr(split_frac=split_frac, cv=cv)       
        

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
    
    def fit_ols(self, split_frac = 0.5):
        
        # Split into training and test sets
        X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=split_frac, random_state=1)

        # Train regression model on training data 
        regr = LinearRegression()
        regr.fit(scale(X_train), scale(y_train))

        # Prediction with test data
        pred = regr.predict(scale(X_test))
        
        return pred
    
    
    def fit_pcr(self, split_frac = 0.5, cv = 10):
        
        # Split into training and test sets
        X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=split_frac, random_state=1)
        
        pca2 = PCA()

        # Scale the data
        X_reduced_train = pca2.fit_transform(scale(X_train))
        n = len(X_reduced_train)

        # 10-fold CV, with shuffle
        kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

        regr = LinearRegression()
        mse = []

        # Calculate MSE with only the intercept (no principal components in regression)
        score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
        mse.append(score)

        # Calculate MSE using CV for the 19 principle components, adding one component at the time.
        for i in np.arange(1, 25):
            score = -1*model_selection.cross_val_score(regr, X_reduced_train[:,:i], y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
            mse.append(score)

        num_pc_to_keep = 20
        X_reduced_test = pca2.transform(scale(X_test))[:,:num_pc_to_keep]

        # Train regression model on training data 
        regr = LinearRegression()
        regr.fit(X_reduced_train[:,:num_pc_to_keep], y_train)

        # Prediction with test data
        pred = regr.predict(X_reduced_test)

        print('MSE is: ', mean_squared_error(y_test, pred))
        print('Test corr is: ',np.corrcoef(y_test,pred)[0,1])
        
        return pred
    
    def fit_pls(self, split_frac = 0.5, cv = 10):
        
        # Split into training and test sets
        X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=split_frac, random_state=1)
    
        # PLS
        # 10-fold CV, with shuffle
        kf_10 = model_selection.KFold(n_splits = cv, shuffle=True, random_state=1)

        mse = []

        for i in np.arange(1, 8):
            pls = PLSRegression(n_components=i)
            score = model_selection.cross_val_score(pls, scale(X_train), y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
            mse.append(-score)

#         # Plot results
#         plt.plot(np.arange(1, 8), np.array(mse), '-v')
#         plt.xlabel('Number of PLS components in regression')
#         plt.ylabel('MSE')

        pls = PLSRegression(n_components=np.argmin(mse))
        pls.fit(scale(X_train), y_train)
        pred = np.squeeze(pls.predict(scale(X_test)))

#         plt.figure()
#         plt.plot(y_test)
#         plt.plot(pred)
#         plt.show()

        print('MSE is: ', mean_squared_error(y_test, pred))
        print('Test corr is: ',np.corrcoef(y_test,pred)[0,1])
        
        return pred
    
    def fit_rrr(self, split_frac = 0.5, cv = 10):
        
        def sqerr(matrix1, matrix2):
            """Squared error (frobenius norm of diff) between two matrices."""
            return around(pow(norm(matrix1 - matrix2, 'fro'), 2) / size(matrix2, 0), 5)

        class ReducedRankRegressor(object):
            """
            Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
            - X is an n-by-d matrix of features.
            - Y is an n-by-D matrix of targets.
            - rrank is a rank constraint.
            - reg is a regularization parameter (optional).
            """
            def __init__(self, X, Y, rank, reg=None):
                if np.size(np.shape(X)) == 1:
                    X = np.reshape(X, (-1, 1))
                if np.size(np.shape(Y)) == 1:
                    Y = np.reshape(Y, (-1, 1))
                if reg is None:
                    reg = 0
                self.rank = rank
                print(X.shape, Y.shape, np.amax(X))

                CXX = np.dot(X.T, X) + reg * sparse.eye(np.size(X, 1))
                CXY = np.dot(X.T, Y)
                _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))
                self.W = V[0:rank, :].T
                self.A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, self.W)).T

            def __str__(self):
                return 'Reduced Rank Regressor (rank = {})'.format(self.rank)

            def predict(self, X):
                """Predict Y from X."""

                if np.size(np.shape(X)) == 1:
                    X = np.reshape(X, (-1, 1))
                return np.dot(X, np.dot(self.A.T, self.W.T))
        
         # Split into training and test sets
        X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=split_frac, random_state=1)
        
        # Fit RRR model
        rank_val = 300
        rr = ReducedRankRegressor(scale(X_train), y_train, rank_val, reg = 1)
        pred = rr.predict(scale(X_test))

#         plt.figure()
#         plt.plot(y_test)
#         plt.plot(pred)
#         plt.show()

        print('MSE is: ', mean_squared_error(y_test, pred))
        print('Test corr is: ',np.corrcoef(y_test,np.squeeze(pred))[0,1])
        
        return pred