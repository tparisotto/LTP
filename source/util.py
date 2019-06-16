#
# Some of the functions are modified versions of methods found in:
# feedforward_neural_network_sparse_input_solution.py
# Author: Barbara Plank (original Keras version). Adapted to PyTorch by Antonio Toral
#


import numpy as np
from sklearn.metrics import accuracy_score,precision_score,balanced_accuracy_score
import torch

def prepare_sequence(data,seq_len):
        data_in_sequence = []
        current_len_seq = 0
        for idx_word in data:
               if(current_len_seq == 0):sequence = []
               sequence.append(idx_word) 
               current_len_seq = current_len_seq + 1
               if(current_len_seq == seq_len): 
                       data_in_sequence.append(sequence)
                       current_len_seq = 0
        return torch.FloatTensor(data_in_sequence)  

def get_list_from_textfile(filepath):
        some_file = open(filepath, "r", encoding="utf8")
        text = some_file.read()
        textlist = text.split()
        return textlist

# Returns shuffled list of tuples (word, {0,1})
def generate_dataset(datapath):
        raw_data = get_list_from_textfile(datapath)
        dataset = []
        for i in range(0, len(raw_data)-1):
                if (raw_data[i] == "<BRK>"): continue
                if (raw_data[i+1] == "<BRK>"):
                        dataset.append((raw_data[i],1))
                else:
                        dataset.append((raw_data[i], 0))
        np.random.shuffle(dataset)     #commentato per fare sequenze di parola da inserire nell'LSTM
        return dataset

def split_dataset(dataset, train_dim=0.8, val_dim=0.1, test_dim=0.1):
        assert(train_dim+val_dim+test_dim == 1)
        dim = len(dataset)
        n_train , n_val = int(dim * train_dim), int(dim * val_dim)
        train, val, test = dataset[:n_train], dataset[n_train:(n_train+n_val)], dataset[(n_train+n_val):]
        return train, val, test


def get_index(word, words2int):
        if word in words2int:
                return words2int[word]
        else:
                return 0

def convert_to_one_hot(word, words2int, vocab_size):
        one_hot = np.zeros(vocab_size)
        one_hot[get_index(word,words2int)] = 1
        return one_hot


def convert_list_to_one_hot(wordlist, words2int, vocab_size):
        out = []
        for word in wordlist:
                one_hot = np.zeros(vocab_size)
                one_hot[get_index(word,words2int)] = 1
                out.append(one_hot)
        return np.array(out)


def tensor_desc(x):
        print("Type:   {}".format(x.type()))
        print("Size:   {}".format(x.size()))
        print("Values: {}".format(x))

# def get_accuracy(y_true, y_pred):
#         assert(len(y_true)==len(y_pred))
#         for i in range(len(y_true)):
#                 if(y_true[i] == )

def get_accuracy(y_true, y_pred):
        assert(len(y_pred)==len(y_true)),"y_true:{},y_pred:{}".format(len(y_true),len(y_pred))
        dim = len(y_pred)
        acc = 0
        for i in range(dim):
                if (y_pred[i] == y_true[i]):
                        acc += 1
        return float(acc / dim)


def eval_accuracy(prediction_path, true_path):
    pred = get_list_from_textfile(prediction_path)
    true = get_list_from_textfile(true_path)

    n_brk_true = 0
    n_correct_brk = 0
    n_errors = 0
    j = 0
    fn = tn = 0

    for i in range(0, len(true)-1):
            if(true[i] == '<BRK>'):
                    n_brk_true += 1
                    if(pred[j] == '<BRK>'):
                            n_correct_brk +=1
                            j += 1
                    #else: fn = fn +1
            else:
                    if(pred[j] == '<BRK>'):
                            n_errors += 1
                            i -= 1
                            j += 1
                    #elif(pred[j] != '<BRK>'): tn = tn +1
                    if (pred[j] != true[i] and pred[j] != true[i+1]):
                        #   --- DEBUG ---
                            print("### Lists Unaligned! Words: ({} , {}) at index j={} , i={} ###".format(pred[j],true[i],j,i)) 
                            print("Context Pred.: {} {} {} {} {}".format(pred[j-3],pred[j-2],pred[j-1],pred[j],pred[j+1]))
                            print("Context True : {} {} {} {} {}".format(true[i-3],true[i-2],true[i-1],true[i],true[i+1]))
                    j += 1
    
    print("Number of breaks in True:                {}".format(n_brk_true))
    print("Number of correctly predicted breaks:    {} out of {}".format(n_correct_brk,n_brk_true))
    print("Number of mistakenly predicted breaks:   {}".format(n_errors))
    print("Model precision score:                    {}".format(n_correct_brk/(n_brk_true + n_errors)))
   # print("Model accuracy score:                    {}".format( (n_correct_brk + tn)/(n_correct_brk + tn + fn + n_errors) ))

def eval_accuracy_2(pred, true):

    n_brk_true = 0
    n_correct_brk = 0
    n_errors = 0


    for i in range(0, len(true)-1):
            if(true[i] == 1):
                    n_brk_true += 1
                    if(pred[i] == 1):
                            n_correct_brk +=1
            else:
                    if(pred[i] == 1):
                            n_errors += 1

    '''
    print("Number of breaks in True:                {}".format(n_brk_true))
    print("Number of correctly predicted breaks:    {} out of {}".format(n_correct_brk,n_brk_true))
    print("Number of mistakenly predicted breaks:   {}".format(n_errors))
    print("Model precision score:                    {}".format(n_correct_brk/(n_brk_true + n_errors)))

   # print("Model accuracy score:                    {}".format( (n_correct_brk + tn)/(n_correct_brk + tn + fn + n_errors) ))'''
    return n_correct_brk/(n_brk_true + n_errors)
def valuation(prediction_path, true_path):
        pred = get_list_from_textfile(prediction_path)
        true = get_list_from_textfile(true_path)

        list_pred = []
        i = 0
        while i < len(pred)-1:
                if(pred[i] != "<BRK>"):
                        if(pred[i+1] == "<BRK>"):
                                i = i + 1
                                list_pred.append(1)
                        else: 
                                list_pred.append(0) 
                i = i + 1
        list_pred.append(0)

        list_true = []
        j = 0
        while j < len(true)-1:
                if(true[j] != "<BRK>"):
                        if(true[j+1] == "<BRK>"):
                                j = j +1
                                list_true.append(1)
                        else: 
                                list_true.append(0) 
                j = j + 1
        list_true.append(0)

        print(balanced_accuracy_score(list_true,list_pred))