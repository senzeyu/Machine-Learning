from __future__ import division
import numpy as np
import csv
import sys
import matplotlib.pyplot as pl
table = {0:'a',1:'e',2:'g',3:'i',4:'l',5:'n',6:'o',7:'r',8:'t',9:'u'}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_forward(a):
    z = np.array([1])#add bias
    for num in a:
        z = np.append(z,sigmoid(num))
    return z

def sigmoid_backward(dl_dz, z):
    z = z[1:]
    dz_da = z * (1-z)
    return dl_dz * dz_da

def softmax_forward(b):
    yhat =np.array([])
    sum = 0.0
    for num in b:
        sum += np.exp(num)
    for num in b:
        yhat=np.append(yhat, np.exp(num) / sum)
    return yhat

def cross_entropy_forward(y, yhat):
    loss = 0.0
    for i in range(len(y)):
        loss -= y[i] * np.log(yhat[i])
    return loss

def cross_entropy_backward(y, yhat):
    return yhat - y

def linear_forward(x, alpha):
    return np.dot(alpha, x)
#a: input, alpha_star: matrix excluding first column for bias, gb: gradient of output
def linear_backward(a,alpha_star, gb):
    alpha_star = np.transpose(alpha_star)
    galpha = np.outer(gb,a)
    ga = np.dot(alpha_star, gb)
    return galpha, ga

def NNforward(x, y, alpha, beta):
    #print('x shape',x.shape)
    #print('alpha shape', alpha.shape)
    a = linear_forward(x, alpha)
    #print('a shape', a.shape)
    z = sigmoid_forward(a)
    #print('z shape', z.shape)
    #print('beta shape', beta.shape)
    b = linear_forward(z, beta)
    #print('b shape', b.shape)
    yhat = softmax_forward(b)
    #print('yhat: ',yhat)
    J = cross_entropy_forward(y, yhat)

    return a,z,b,yhat,J

def NN_backward(x, y, alpha, beta, yhat, z):
    gb = cross_entropy_backward(y, yhat)
    #print('gb: ',gb.shape)
    gbeta, gz = linear_backward(z, beta[:,1:],gb)
    #print('gbeta: ', gbeta.shape)
    #print('gz: ', gz.shape)
    ga = sigmoid_backward(gz, z)
    #print('ga: ', ga.shape)
    galpha, gx = linear_backward(x,alpha[:,1:],ga)
    return galpha, gbeta

def read_input(input):
    y = np.array([])
    x = []
    with open(input, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            y=np.append(y,int(row[0]))
            tmpx = np.array([1]) #add bias
            for i in row[1:]:
                tmpx = np.append(tmpx,int(i))
            x.append(tmpx)
    x = np.array(x)
    return x,y

train_cross_entropy = []
test_cross_entropy = []
def train(train_in, epoch, hidden_units, init_flag,learning_rate,test_in):
    train_cross_entropy = []
    test_cross_entropy = []
    x, y = read_input(train_in)
    test_x, test_y = read_input(test_in)
    M = x.shape[1]
    D = hidden_units
    K = 10
    result_string = ''
    #alpha: D X (M + 1), beta: K X (D + 1)
    if init_flag == 1:
        alpha = np.random.uniform(-0.1, 0.1,(D,M))
        beta = np.random.uniform(-0.1, 0.1,(K,D+1))
        #initialize bias terms to 0
        alpha[:,0] = 0.0
        beta[:,0] = 0.0
    else:
        alpha =np.zeros((D,M))
        beta = np.zeros((K,D+1))
    for i in range(epoch):
        for j in range(len(y)):
            #print(y[j])
            vec_y = np.zeros(10)#make y a vector
            vec_y[int(y[j])] = 1.0
            a, z, b, y_hat, J = NNforward(np.transpose(x[j]),vec_y,alpha,beta)
            galpha, gbeta = NN_backward(np.transpose(x[j]),vec_y,alpha,beta,y_hat,z)
            #print(galpha.shape)
            #print(alpha.shape)
            alpha = alpha - learning_rate * galpha
            beta = beta - learning_rate * gbeta
        #evaluate training mean cross-entropy
        train_mce = mean_cross_entropy(x, y, alpha, beta)
        result_string += 'epoch='+str(i+1)+' crossentropy(train): '+str(train_mce) + '\n'
        #evaluate test mean cross-entropy
        test_mce = mean_cross_entropy(test_x, test_y, alpha, beta)
        result_string += 'epoch=' + str(i + 1) + ' crossentropy(test): ' + str(test_mce) + '\n'
        train_cross_entropy.append(train_mce)
        test_cross_entropy.append(test_mce)

    pl.plot(range(epoch), train_cross_entropy, label="Train")
    pl.plot(range(epoch), test_cross_entropy, label="Test")
    pl.xlabel("#epoch")
    pl.ylabel("Mean Cross Entropy")
    pl.legend(loc='upper right')
    pl.show()
    return alpha, beta, result_string, train_cross_entropy, test_cross_entropy

def mean_cross_entropy(x, y, alpha, beta):
    entropy = 0.0
    for i in range(len(y)):
        vec_y = np.zeros(10)
        vec_y[int(y[i])] = 1.0
        a, z, b, y_hat, J = NNforward(np.transpose(x[i]), vec_y, alpha, beta)
        entropy += J
    return entropy / len(y)

def predict(test_in, alpha, beta):
    x,y = read_input(test_in)
    yh = []
    for i in range(len(y)):
        vec_y = np.zeros(10)
        vec_y[int(y[i])] = 1.0
        a, z, b, y_hat, J = NNforward(np.transpose(x[i]), vec_y, alpha, beta)
        yh.append(np.argmax(y_hat))
    yh = np.array(yh)

    err = np.sum(yh != y) / len(y)
    print(err)
    return yh,err

if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)
    train_in = args[0]
    test_in = args[1]
    train_out = args[2]
    test_out = args[3]
    metrics_out = args[4]
    epoch = int(args[5])
    hidden_units = int(args[6])
    init_flag = int(args[7])
    learning_rate = float(args[8])

    # learning_rate = 0.01
    # epoch = 100
    # init_flag = 1
    # hidden_units = [5,20,50,100,200]
    #
    # trainCE = []
    # testCE = []
    # for unit in hidden_units:
    #     alpha, beta, result_string,avg_train,avg_test = train(train_in, epoch, unit, init_flag, learning_rate, test_in)
    #     trainCE.append(sum(avg_train)/len(avg_train))
    #     testCE.append(sum(avg_test) / len(avg_test))
    #
    # pl.plot(hidden_units, trainCE, label="Train")
    # pl.plot(hidden_units, testCE, label="Test")
    # pl.xlabel("Hidden Units")
    # pl.ylabel("Average Cross Entropy")
    # pl.legend(loc='upper right')
    # pl.show()

    learning_rate = 0.1
    epoch = 100
    init_flag = 1
    hidden_units = 50
    alpha, beta, result_string,avg_train,avg_test = train(train_in, epoch, hidden_units, init_flag, learning_rate, test_in)



    # alpha, beta, result_string = train(train_in,epoch,hidden_units,init_flag,learning_rate,test_in)
    #
    # yh1, err1 = predict(train_in, alpha, beta)
    # with open(train_out, "w") as f:
    #     for label in yh1:
    #         f.write(str(int(label)) + "\n")
    #
    # yh2,err2 = predict(test_in,alpha,beta)
    # with open(test_out, "w") as f:
    #     for label in yh2:
    #         f.write(str(int(label)) + "\n")
    #
    # with open(metrics_out, "w") as metric:
    #     result_string += 'error(train): ' + str(err1) + '\n'
    #     result_string += 'error(test): ' + str(err2) + '\n'
    #     metric.write(result_string)