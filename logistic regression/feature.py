# mode 1:
# word of the dictionary occurs at least once in the movie review
# mode 2:
# word of the dictionary occurs (1 - 4) times in the review
import sys

def build_dic(dict_in):
    dictionary = {}
    for line in dict_in:
        info = line.strip().split()
        #eg dictionary[films] = 0
        dictionary[info[0]] = info[1]
    return dictionary

def model1(infile, outfile,dictionary):
    for line in infile:
        line = line.strip().split("\t")
        label = line[0]
        words = line[1].split(" ")
        records = [] #storing indices of words that has appeared
        for word in words:
            if word in dictionary.keys():#optimize?
                if dictionary[word] not in records:
                    records.append(dictionary[word])
        result = ''
        result += label
        for record in records:
             result += ("\t"+record+":1")
        result += "\n"
        outfile.write(result)
        #print(result)
    return

def model2(infile, outfile,dictionary):
    for line in infile:
        line = line.strip().split("\t")
        label = line[0]
        words = line[1].split(" ")
        records = [] #storing indices of words that has appeared
        count = {}
        for word in words:
            if word in dictionary.keys():#optimize?
                tag = dictionary[word]
                if tag not in records:
                    records.append(tag)
                if tag not in count.keys():
                    count[tag] = 1
                else:
                    count[tag] += 1
        result = ''
        result += label
        for record in records:
            if count[record] < 4:
                result += ("\t"+record+":1")
        result += "\n"
        #print(result)
        outfile.write(result)
    return

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    train_in = open(args[0],"r")
    valid_in = open(args[1], "r")
    test_in = open(args[2], "r")
    dic_file = open(args[3], "r")
    train_out = open(args[4], "w")
    valid_out = open(args[5], "w")
    test_out = open(args[6], "w")
    flag = int(args[7])

    dictionary = build_dic(dic_file)
    if flag == 1:
        model1(train_in,train_out,dictionary)
        model1(valid_in, valid_out, dictionary)
        model1(test_in, test_out, dictionary)
    elif flag == 2:
        model2(train_in,train_out,dictionary)
        model2(valid_in, valid_out, dictionary)
        model2(test_in, test_out, dictionary)
