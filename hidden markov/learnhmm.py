import sys
import numpy as np
tag_dict = {}
word_dict = {}

def read_trainwords(train_input):
    tag_ct = len(tag_dict)
    word_ct = len(word_dict)
    # +1  is added to make a pseudocount
    init_ct = np.ones(tag_ct)
    emit_ct = np.ones((tag_ct, word_ct))
    trans_ct = np.ones((tag_ct,tag_ct))
    with open(train_input, 'r') as f:
        a = 0
        for line in f.readlines()[0:10000]:
            # print(a)
            a+=1
            #initial
            sentence = line.strip().split(' ')
            first_word,first_tag = sentence[0].split('_')
            init_ct[tag_dict[first_tag]] += 1
            state = []
            #emission
            for pair in sentence:
                word,tag = pair.split('_')
                emit_ct[tag_dict[tag]][word_dict[word]] += 1
                state.append(tag_dict[tag])
            #transmission
            for i in range(len(state) - 1):
                trans_ct[state[i]][state[i + 1]] += 1
    return init_ct, emit_ct, trans_ct

def read_index2word(index_to_word):
    ind = 0
    with open(index_to_word, 'r') as f:
        for line in f:
            word_dict[line.strip()] = ind
            ind += 1

def read_index2tag(index_to_tag):
    ind = 0
    with open(index_to_tag, 'r') as f:
        for line in f:
            tag_dict[line.strip()] = ind
            ind += 1

def est_initialization(init_ct):
    init_prob = np.ones(len(tag_dict))
    total = np.sum(init_ct)
    for i in range(len(tag_dict)):
        init_prob[i] = init_ct[i] / total
    return init_prob

def est_transition(trans_ct):
    trans_prob = np.ones((len(tag_dict), len(tag_dict)))
    for i in range(len(tag_dict)):
        total = np.sum(trans_ct[i])
        for j in range(len(tag_dict)):
            trans_prob[i][j] = trans_ct[i][j] / total
    return trans_prob

def est_emissition(emit_ct):
    emit_prob = np.ones((len(tag_dict), len(word_dict)))
    for i in range(len(tag_dict)):
        total = np.sum(emit_ct[i])
        for j in range(len(word_dict)):
            emit_prob[i][j] = emit_ct[i][j] / total
    return emit_prob

if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    read_index2tag(index_to_tag)
    read_index2word(index_to_word)
    # print(tag_dict)
    # print(word_dict)
    # print(word_dict['AUSTRIA'])
    init_ct, emit_ct, trans_ct = read_trainwords(train_input)
    init_prob = est_initialization(init_ct)
    emit_prob = est_emissition(emit_ct)
    trans_prob = est_transition(trans_ct)
    #print(init_prob)
    with open(hmmprior,'w') as f:
        for i in range(len(tag_dict)):
            f.write('{:.20E}'.format(init_prob[i]) + '\n')
    with open(hmmemit,'w') as f:
        for i in range(len(tag_dict)):
            tmp = ''
            for j in range(len(word_dict)):
                tmp += '{:.20E}'.format(emit_prob[i][j])  + ' '
                #tmp += str(emit_prob[i][j]) + ' '
            f.write(tmp[:-1] + '\n')
    with open(hmmtrans,'w') as f:
        for i in range(len(tag_dict)):
            tmp = ''
            for j in range(len(tag_dict)):
                tmp += '{:.20E}'.format(trans_prob[i][j]) + ' '
                #tmp += str(trans_prob[i][j]) + ' '
            f.write(tmp[:-1] + '\n')