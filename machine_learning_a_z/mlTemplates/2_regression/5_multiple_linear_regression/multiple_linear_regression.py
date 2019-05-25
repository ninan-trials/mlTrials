# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:,-1] = labelencoder_x.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [-1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm

X = np.append(np.ones((50, 1)).astype(int), X, 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = backwardElimination(X_opt, SL)

X_opt_train, X_opt_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)
regressor.fit(X_opt_train, y_train)
y_pred_belim = regressor.predict(X_opt_test)

"""

In [42]: y_pred_belim
Out[42]:
array([102284.64605183, 133873.92383812, 134182.1495165 ,  73701.1069363 ,
       180642.25299736, 114717.24903894,  68335.07575312,  97433.45922275,
       114580.92136452, 170343.31979498])
       
       
Out[49]: y_pred_belim
array([104667.27805998, 134150.83410578, 135207.80019517,  72170.54428856,
       179090.58602508, 109824.77386586,  65644.27773757, 100481.43277139,
       111431.75202432, 169438.14843539])

In [43]: y_test
Out[43]:
array([103282.38, 144259.4 , 146121.95,  77798.83, 191050.39, 105008.31,
        81229.06,  97483.56, 110352.25, 166187.94])

In [44]: y_pred
Out[44]:
array([103015.20159796, 132582.27760816, 132447.73845175,  71976.09851259,
       178537.48221054, 116161.24230163,  67851.69209676,  98791.73374688,
       113969.43533012, 167921.0656955 ])


In [46]: y_pred_ols
Out[46]:
array([103125.01275975, 134638.87007529, 135011.91472396,  74113.88870454,
       181405.37809703, 114978.60515008,  68631.3183233 ,  98314.54885378,
       114990.38463925, 171127.62321762])
"""
