import numpy as np

def cost_function(X,Y,w,b):
    cost = 0
    for i in range(size):
        f_wb = np.dot(X[i],w) + b
        cost+=(f_wb - Y[i])**2
    total_cost = cost/(2*size)
    return total_cost

def gradient(X,Y,w,b):
    m,n = X.shape    # m = training examples , n = features
    dJ_dw = np.zeros(n)
    dJ_db = 0
    for i in range(m):
        error = (np.dot(X[i],w) + b) - Y[i]
        for j in range(n):
            dJ_dw = dJ_dw + error*X[i,j]
        dJ_db+=error
    dJ_dw/=m
    dJ_db/=m
    return dJ_dw,dJ_db

def gradient_descent(X,Y,w,b,alpha,iter):
    w_values = []
    b_values = []
    cost_values = []
    iter_values = []
    for i in range(iter):
        dJ_dw,dJ_db = gradient(X,Y,w,b)
        w = w - alpha*dJ_dw
        b = b - alpha*dJ_db
        cost = cost_function(X,Y,w,b)
        w_values.append(w)
        b_values.append(b)
        cost_values.append(cost)
        iter_values.append(i)
    return w_values,b_values,cost_values,iter_values


        



X = np.asarray([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
Y = np.asarray([460, 232, 178])
size = X.shape[0]
w = np.zeros(4)
b = 0.
fw,fb,fcost,fiter = gradient_descent(X,Y,w,b,5.0e-7,2000)
for i in range((len(fw)//100)):
    print(f"w,b = {fw[i*100]},{fb[i*100]} ; cost = {fcost[i*100]} ; iteration = {fiter[i*100]}")

