from __future__ import print_function
from __future__ import division
import sys
import numpy as np
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

def H(X):#X is a list
    arr, ct = own_unique(X)
    total = sum(ct)
    result = 0
    for case in ct:
        print(case/total)
        result -= case/total * np.log2(case/total)
    #print(result)
    return result

def majorVote(Y):
    arr, ct = own_unique(Y)
    #print(arr)
    #print(ct)
    val = arr[np.argmax(ct)]
    return val

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]

    dataset,output = create(infile)
    entropy = H(output)

    val = majorVote(output)
    err = 0
    for o in output:
        if not o == val:
            err += 1
    err /= len(output)

    with open(outfile,'w') as out:
        out.write('entropy: '+str(entropy)+'\n')
        out.write('error: '+str(err))

