import numpy as np
import math
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_forward(a):
    z = np.array([1])
    for num in a:
        z = np.append(z,sigmoid(num))
    return z

def sigmoid_backward(dl_dz, z):
    z = z[1:]
    dz_da = z * (1-z)
    dl_da = dl_dz * dz_da
    return dl_da

def softmax_forward(b):
    yhat =np.array([])
    sum = 0.0
    for num in b:
        sum += math.exp(num)
    for num in b:
        yhat=np.append(yhat, math.exp(num) / sum)
    return yhat

def cross_entropy_forward(y, yhat):
    loss = 0.0
    for i in range(len(y)):
        loss -= y[i] * math.log(yhat[i])
    return loss

def cross_entropy_backward(y, yhat):
    # dl_db = np.array([])
    # for i in range(len(y)):
    #     dl_db = np.append(dl_db, float(yhat[i] - y[i]))
    # return dl_db
    return yhat-y

def linear_forward(x, alpha):
    return np.dot(alpha, x)

#a: input, alpha_star: matrix excluding first column for bias, gb: gradient of output
def linear_backward(a,alpha_star, gb):
    alpha_star = np.transpose(alpha_star)
    galpha = np.outer(gb,a)
    ga = np.dot(alpha_star, gb)
    return galpha, ga

def NNforward(x, y, alpha, beta):
    a = linear_forward(x, alpha)
    z = sigmoid_forward(a)
    b = linear_forward(z, beta)
    y_hat = softmax_forward(b)
    J = cross_entropy_forward(y, y_hat)

    return a,z,b,y_hat,J

def NN_backward(x, y, alpha, beta, yhat, z):
    gb = cross_entropy_backward(y, yhat)
    gbeta, gz = linear_backward(z, beta[:,1:],gb)
    ga = sigmoid_backward(gz, z)
    galpha, gx = linear_backward(x,alpha[:,1:],ga)
    return galpha, gbeta

if __name__ == "__main__":
    x = np.transpose(np.array([1,1,1,0,0,1,1]))
    alpha = np.array([[1,1,2,-3,0,1,-3],
                      [1,3,1,2,1,0,2],
                      [1,2,2,2,2,2,1],
                      [1,1,0,2,1,-2,2]])
    alpha_star = alpha[:,1:]
    beta = np.array([[1,1,2,-2,1],
                     [1,1,-1,1,2],
                     [1,3,1,-1,1]])
    beta_star = beta[:,1:]

    y = np.array([0,1,0])
    #forward
    a = linear_forward(x,alpha)
    print("a: ",a)
    z = sigmoid_forward(a)
    print("z: ",z)
    b = linear_forward(z,beta)
    print("b: ",b)
    yhat = softmax_forward(b)
    print("yhat: ",yhat)
    loss = cross_entropy_forward(y,yhat)
    print("loss: ",loss)
    #backward
    dl_db = cross_entropy_backward(y,yhat)
    print("dl/db: ",dl_db,"\n")
    dl_dbeta, dl_dz = linear_backward(z,beta_star,dl_db)
    print("dl/d(beta): ")
    print(dl_dbeta,"\n")
    print("dl/dz: ")
    print(dl_dz,"\n")
    dl_da = sigmoid_backward(dl_dz,z)
    print("dl/da: ")
    print(dl_da, "\n")
    dl_dalpha, dl_dx = linear_backward(x,alpha_star,dl_da)
    print("dl/d(alpha): ")
    print(dl_dalpha, "\n")

    updated_alpha = alpha - dl_dalpha
    print("updated alpha: ")
    print(updated_alpha, "\n")

    print("dl/dx: ")
    print(dl_dx, "\n")

    a,b = NN_backward(x,y,alpha,beta,yhat,z)
    print(a)
    print(b)



