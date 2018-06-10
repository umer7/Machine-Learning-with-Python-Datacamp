# The squared error, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        predicted_y_i = w@X[i]
        s = s + (predicted_y_i - y[i])**2
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LinearRegression
lr = LinearRegression(fit_intercept=False).fit(X,y)
print(lr.coef_)
