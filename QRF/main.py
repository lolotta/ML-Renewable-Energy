from quantile_forest import RandomForestQuantileRegressor
from sklearn import datasets
X, y = datasets.fetch_california_housing(return_X_y=True)
qrf = RandomForestQuantileRegressor()
qrf.fit(X, y)
y_pred = qrf.predict(X, quantiles=[0.025, 0.5, 0.975])
print(y_pred)