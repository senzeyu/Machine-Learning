from __future__ import print_function
from __future__ import division
import numpy as np
import sys
#from matplotlib import pyplot

def create(infile):#create dataset, and output based on infile
    lines = open(infile).readlines()
    dataset = {}
    names = []
    infos = []
    output = []
    for line in lines:
        if line == lines[0]:
            names = line.split()
            for i in range(len(names) - 1):
                dataset[names[i]] = []

        else:
            infos = line.split()
            for i in range(len(infos)):
                if i == len(infos) - 1:
                    output.append(infos[i])
                else:
                    dataset[names[i]].append(infos[i])
    #print(dataset)
    #print(output)
    return dataset, output

def own_unique(X):
    unique_elements = []
    counts = []
    for x in X:
        if x in unique_elements:
            counts[unique_elements.index(x)] += 1
        else:
            unique_elements.append(x)
            counts.append(1)
    return unique_elements, counts


def split(X):#X is a list, split into a dictionary {binary value : index}
    vals,ct = own_unique(X)
    dict = {}
    for val in vals:
        dict[val] = []
    for ct in range(len(X)):
        dict[X[ct]].append(ct)
    #print(dict)
    return dict

def H(X):#X is a list
    arr, ct = own_unique(X)
    total = sum(ct)
    result = 0
    for case in ct:
        result -= case/total * np.log2(case/total)
    #print(result)
    return result

def I(X,Y):
    arr, ct = own_unique(X)
    #print(arr)
    #print(ct)
    total = sum(ct)
    HYgivenX = 0
    for i in range(len(arr)):
        tmplist = []
        tmpind = 0
        for x in X:
            if x == arr[i]:
                tmplist.append(Y[tmpind])
            tmpind += 1
        # HYgivenX += P(X = arr[i]) * H( Y | X = arr[i] )
        HYgivenX += (ct[i] / total) * H(tmplist)
    result = H(Y) - HYgivenX
    #print(result)
    return result

def majorVote(Y):
    arr, ct = own_unique(Y)
    #print(arr)
    #print(ct)
    val = arr[np.argmax(ct)]
    return val

def create_subdataset(dataset, indices, removekey):
    newdataset = {}
    for key in dataset:
        newdataset[key] = []
    for key in dataset:
        for ind in indices:
            newdataset[key].append(dataset[key][ind])
    newdataset.pop(removekey, None)
    return newdataset

def create_subout(out, indices):
    newout = []
    for ind in indices:
        newout.append(out[ind])
    return newout

def recur(dataset, out, curdepth, maxdepth):
    #print(curdepth, maxdepth)
    if maxdepth == 0:
        return majorVote(out)
    if(len(out) == 0) or (len(set(out)) == 1):
        return out[0]
    if curdepth == maxdepth:
        return majorVote(out)
    if len(dataset.keys()) == 0:#can't be further splitted
        return majorVote(out)
    maxgain = 0
    for key in dataset:
        gain = I(dataset[key], out)
        if gain > maxgain:
            maxgain = gain
            maxkey = key
    if(maxgain == 0):
        return majorVote(out)
    descendents = split(dataset[maxkey])
    curdepth += 1
    branch = {}
    for val, indices in descendents.items():
        sub_dataset = create_subdataset(dataset, indices, maxkey)
        sub_out = create_subout(out, indices)
        branch[maxkey+'='+val]=recur(sub_dataset,sub_out,curdepth,maxdepth)

    return branch

def predict(dataset,tree):
    result = []
    for key in dataset:
        length = len(dataset[key])
        break
    for i in range(length):
        if not type(tree) is dict:
            result.append(tree)
        else:
            subtree = tree
            tmpresult = {}
            while type(tmpresult) is dict:
                for choice in subtree:
                    name=choice.split('=')[0]
                tmpresult = subtree[name+'='+dataset[name][i]]
                subtree = subtree[name+'='+dataset[name][i]]
            result.append(tmpresult)
    return result

def draw(tree, depth):
    for key in tree:
        print('| '*depth + key, end='')
        subtree = tree[key]
        if type(subtree) is dict:
            print()
            draw(subtree, depth + 1)
        else:
            print(' : '+subtree)





if __name__=="__main__":
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    maxdepth = int(sys.argv[3])
    trainoutfile = sys.argv[4]
    testoutfile = sys.argv[5]
    metricsout = sys.argv[6]

    trainset, trainout = create(trainfile)
    testset, testout = create(testfile)
    print(len(trainset))

    trainerrors = []
    testerrors = []
    for i in range(len(trainset)+1):
        tree = recur(trainset, trainout, 0, i)
        predictout = predict(trainset, tree)
        predictout2 = predict(testset,tree)

        train_err = 0
        for j in range(len(predictout)):
            if predictout[j] != trainout[j]:
                train_err += 1
        train_err /= len(predictout)
        trainerrors.append(train_err)

        test_err = 0
        for j in range(len(predictout2)):
            if predictout2[j] != testout[j]:
                test_err += 1
        test_err /= len(predictout2)
        testerrors.append(test_err)
    print(trainerrors)
    print(testerrors)

    x = np.arange(0, len(trainset)+1, 1)

    print(x)

    #use tree on training data
    #print(tree)
    #draw(tree,1)

    #print(predictout)
    with open(trainoutfile,'w') as out:
        for x in predictout:
            out.write(x+'\n')
    out.close()

    #use tree on testing data
    print(tree)
    predictout = predict(testset, tree)
    #print(predictout)
    with open(testoutfile,'w') as out:
        for x in predictout:
            out.write(x+'\n')
    out.close()

    test_err = 0
    for i in range(len(testout)):
        if predictout[i] != testout[i]:
            test_err += 1
    test_err /= len(testout)

    with open(metricsout,'w') as out:
        out.write('error(train): '+str(train_err)+'\n')
        out.write('error(test): '+str(test_err)+'\n')
    out.close

