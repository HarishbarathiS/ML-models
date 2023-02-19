import numpy as np

def cost_function(X,Y,w,b):
    
    cost = 0
    for i in range(size):
        cost+=((w*X[i]+b) - Y[i])**2
    total_cost = cost/2*size
    return total_cost


def gradient(X,Y,w,b):
    dJ_dw = 0
    dJ_db = 0
    for i in range(size):
        dJ_dw+=(w*X[i]+b - Y[i])*X[i]
        dJ_db+=(w*X[i]+b - Y[i])
    dJ_dw/=size
    dJ_db/=size
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

    
X = np.asarray([1,2])
Y = np.asarray([300,500])
size = X.shape[0]
fw,fb,fcost,fiter = gradient_descent(X,Y,0,0,0.1,907)
for i in range(len(fw)):
    print(f"w,b = ({fw[i]},{fb[i]}) ; iteration = {fiter[i]} ; cost function = {fcost[i]}")





