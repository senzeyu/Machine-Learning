import numpy as np
import sys
import matplotlib.pyplot as pl
def build_dic(dict_in):
    dictionary = {}
    for line in dict_in:
        info = line.strip().split()
        #eg dictionary[films] = 0
        dictionary[info[0]] = info[1]
    return dictionary

def read_formatted(infile):
    data = []
    labels = []
    for line in infile:
        line = line.strip().split("\t")
        labels.append(int(line[0]))
        #print(line[1:])
        tmp = {}
        for tag_val in line[1:]:
            tag_val = tag_val.split(":")
            tmp[int(tag_val[0])] = int(tag_val[1])
            #data.append({tag_val[0] : tag_val[1]})
        data.append(tmp)
    #print(len(labels))
    #print(len(data))
    return data, labels

def sigmoid(x):
    #return np.exp(x) / (1.0 + np.exp(x))
    return 1.0/(1 + np.exp(-x))


def inner_SGD(theta, learning_rate, x, y):
    product = 0.0
    for key in x.keys():
        product += theta[key]#weight
    product += theta[-1]#intercept

    y_hat = sigmoid(product)
    update = learning_rate * (y - y_hat)
    for key in x.keys():
        theta[key] += update #weight, x is always 1
    theta[-1] += update#intercept
    return theta

train_likelihood = []
valid_likelihood = []
def train(dictionary, epoch, learning_rate, data, labels, valid_data, valid_labels):
    #fastest way to initialize list:
    #https://www.geeksforgeeks.org/python-which-is-faster-to-initialize-lists/

    theta = [0.0] * (len(dictionary))
    for i in range(epoch):
        for j in range(len(data)):
            theta = inner_SGD(theta, learning_rate, data[j], labels[j])
        train_likelihood.append(likelihood(theta, data, labels))
        valid_likelihood.append(likelihood(theta, valid_data, valid_labels))
    return theta

def predict(theta, infile, outfile):
    #print(infile)
    data, actual_labels = read_formatted(infile)
    #print(len(data))
    #print(len(actual_labels))
    predict_labels = []
    for i in range(len(data)):
        product = 0.0
        for key in data[i].keys():
            product += theta[key]
        product += theta[-1]
        if sigmoid(product) >= 0.5:
            predict_labels.append(1)
            outfile.write('1\n')
        else:
            predict_labels.append(0)
            outfile.write('0\n')
    #calculate error
    err = 0.0
    for i in range(len(actual_labels)):
        if actual_labels[i] != predict_labels[i]:
            err += 1.0

    #print('err num: '+str(err))
    #print('total: '+str(len(actual_labels)))
    return err / float(len(actual_labels))

def likelihood(theta, x, y):
    result = 0.0
    for i in range(len(y)):
        product = 0.0
        for key in x[i].keys():
            product += theta[key]
        product += theta[-1]
        result += (-y[i] * product + np.log(1+np.exp(product)))
    return result / len(y)

if __name__ == '__main__':
    args = sys.argv[1:]
    #print(args)
    train_in = open(args[0], "r")
    valid_in = open(args[1], "r")
    test_in = open(args[2], "r")
    dic_file = open(args[3], "r")
    train_out = open(args[4], "w")
    test_out = open(args[5], "w")
    metrics_out = open(args[6], "w")
    epoch = int(args[7])

    dictionary = build_dic(dic_file)
    data,labels = read_formatted(train_in)
    valid_data, valid_labels = read_formatted(valid_in)
    theta = train(dictionary, epoch, 0.1, data, labels,valid_data,valid_labels)


    #print(theta)
    train_in.close()
    train_in = train_in = open(args[0], "r")
    train_error = predict(theta, train_in, train_out)
    test_error = predict(theta, test_in, test_out)
    print('error(train): '+str(train_error) + "\n")
    print('error(test): ' + str(test_error) + "\n")
    metrics_out.write('error(train): '+str(train_error) + "\n")
    metrics_out.write('error(test): ' + str(test_error) + "\n")

    x = np.arange(0, epoch)
    # print(x)
    # for i in range(len(train_likelihood)):
    #     train_likelihood[i] /= 1000
    #     valid_likelihood[i] /= 1000
    # print(train_likelihood)
    # print(valid_likelihood)


    pl.plot(x, train_likelihood, label = "Train")
    pl.plot(x, valid_likelihood, label = "Validation")
    pl.xlabel("#Epoch")
    pl.ylabel("Negative log-likelihood")
    pl.legend(loc='upper right')
    pl.show()