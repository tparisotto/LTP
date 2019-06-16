import util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.metrics import accuracy_score,precision_score,balanced_accuracy_score
from sklearn.utils import class_weight

from collections import Counter

data_path = "./data/preprocessed_all.txt"

train_dim       = 0.8
val_dim         = 0.1
test_dim        = 0.1 
num_epochs      = 20
size_batch      = 32
embedding_dim   = 32
lstm_dim        = 64
seq_len         = 12

# Edited from feedforward_neural_network_sparse_input.py
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, lstm_dim, output_dim,size_batch):
        super(RNN, self).__init__()
        self.size_batch = size_batch
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_dim)
        self.hidden = nn.Linear(lstm_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prev_state):
        x_embed = self.embedding(x.long())
        lstm_out, state = self.lstm(x_embed ,prev_state)
        x_hidden = self.hidden(lstm_out)
        y = self.sigmoid(x_hidden)
        return y, state

def load_data(data_path):
    dataset = util.generate_dataset(data_path)
    train, val, test = util.split_dataset(dataset, train_dim, val_dim, test_dim)
    words = []
    labels = []
    for w,l in dataset:
        words.append(w)
        labels.append(l)
    vocabulary = sorted(set(words))
    words2int = {w: i for i, w in enumerate(vocabulary)}
    int2words = {k: w for k, w in enumerate(words2int)}

    vocab_size  = len(vocabulary)
    num_classes = 1
    print("Vocabulary Size:     {}".format(vocab_size))
    print("Number of Classes:   {}".format(num_classes))

    return train, val, test, words, labels, vocabulary, vocab_size, num_classes, words2int, int2words

print("Loading Data...")
train, val, test, words, labels, vocabulary, vocab_size, num_classes, words2int, int2words = load_data(data_path)

model       = RNN(input_dim=vocab_size, output_dim=num_classes, embedding_dim=embedding_dim, lstm_dim=lstm_dim, size_batch = size_batch)
criterion   = nn.BCEWithLogitsLoss(reduce = 'none')
class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
class_weights = torch.FloatTensor(class_weights)
optimizer   = optim.Adam(params=model.parameters())

print("Model: {}".format(model))
print("Loss criterion: {}".format(criterion))
print("Optimizer: {}".format(optimizer))

print("##### Training #####")
num_batches = len(train) // size_batch
print("Batch size: {}, Number of batches: {}".format(size_batch, num_batches))

for epoch in range(num_epochs):
    print("Epoch: {}/{}".format(epoch+1, num_epochs))
    epoch_loss = 0
    h0, c0 = Variable(torch.zeros(1, seq_len, lstm_dim)), Variable(torch.zeros(1, seq_len, lstm_dim))
    for batch in range(num_batches):
        # if((batch+1) % 50 == 0 or (batch+1) == num_batches): 
        #     print("Training on batch: {}/{}".format(batch+1,num_batches))
        batch_begin = batch*size_batch
        batch_end = (batch+1)*size_batch*seq_len
        if(batch_end > len(labels)):
            batch_end = len(labels)
        x_data = []
        y_data = []
        for i in range(batch_begin, batch_end):
            #x_data.append(util.convert_to_one_hot(words[i], words2int, vocab_size)) #caso one-hot
            x_data.append(util.get_index(words[i], words2int))  #caso indice per parola
            y_data.append(labels[i])


        x_tensor = util.prepare_sequence(x_data,seq_len)
        y_tensor = util.prepare_sequence(y_data,seq_len)
        #x_tensor = Variable(torch.FloatTensor(seq))
        #y_tensor = Variable(torch.FloatTensor(y_data))

        optimizer.zero_grad()

        y_prob, (hn, cn) = model(x_tensor, (h0, c0))
        #util.tensor_desc(y_prob)
        #print(y_tensor.shape)
        y_prob = torch.squeeze(y_prob)
        y_prob = y_prob.view(-1, seq_len)
        loss = criterion(y_prob, y_tensor)
        loss = torch.mul(class_weights, loss )               
        loss= torch.mean(loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("Average Loss: {}".format(epoch_loss/num_batches))


    print(" Validation: ")
    epoch_acc = epoch_prec = 0
    num_batches_val = len(val) // size_batch
    print("Number of batches for validation: {}".format(num_batches_val))
    for batch in range(num_batches_val):
        batch_begin = batch*size_batch
        batch_end = (batch+1)*size_batch*seq_len

        if(batch_end > len(labels)):
            batch_end = len(labels)
        x_data_val = []
        y_data_val = []
        for i in range(batch_begin, batch_end):
            #x_data_val.append(util.convert_to_one_hot(words[i], words2int, vocab_size))
            x_data_val.append(util.get_index(words[i], words2int))  #caso indice per parola
            y_data_val.append(labels[i])

    #x_tensor_val = Variable(torch.FloatTensor(x_data))
    #y_tensor_val = Variable(torch.FloatTensor(y_data))
    x_tensor_val = util.prepare_sequence(x_data_val,seq_len)
    y_tensor_val = util.prepare_sequence(y_data_val,seq_len)

    y_pred_val,_ = model(x_tensor_val, (hn ,cn))
    #print(y_pred_val.shape)
    y_pred_val = torch.squeeze(y_pred_val,2)
    #output = torch.max(y_pred_val, dim=1)  #non serve perché ogni cella dell'lstm manda in output la label predetta
    new_y_pred_val = torch.round(y_pred_val)

    #costruzione delle due liste per calcolo accuracy
    prediction = []
    true = []
    new_y_pred_val_np = new_y_pred_val.detach().numpy()
    new_y_true_val_np = y_tensor_val.detach().numpy()

    for row in new_y_pred_val_np:
        for value in row:
            if (value == 0): prediction.append(0)
            else: prediction.append(1)
    
    for row in new_y_true_val_np:
        for value in row:
            if (value == 0): true.append(0)
            else: true.append(1)

    epoch_acc += balanced_accuracy_score(true, prediction)
    epoch_prec = precision_score(true, prediction)
    print("Epoch Accuracy: {}".format(epoch_acc))
    print("Epoch Precision: {}".format(epoch_prec))
