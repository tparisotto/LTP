import util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.utils import class_weight


from collections import Counter

data_path = "./data/preprocessed_all.txt"

train_dim       = 0.8
val_dim         = 0.1
test_dim        = 0.1 
num_epochs      = 200
size_batch      = 32
embedding_dim   = 64
lstm_dim        = 64


# Edited from feedforward_neural_network_sparse_input.py
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, lstm_dim, output_dim):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True)
        self.hidden = nn.Linear(lstm_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prev_state):
        x_embed = self.embedding(x.long())
        lstm_out, state = self.lstm(x_embed, prev_state)
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

model       = RNN(input_dim=vocab_size, output_dim=num_classes, embedding_dim=embedding_dim, lstm_dim=lstm_dim)
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
    h0, c0 = Variable(torch.zeros(1, size_batch, lstm_dim)), Variable(torch.zeros(1, size_batch, lstm_dim))
    for batch in range(num_batches):
        # if((batch+1) % 50 == 0 or (batch+1) == num_batches): 
        #     print("Training on batch: {}/{}".format(batch+1,num_batches))
        batch_begin = batch*size_batch
        batch_end = (batch+1)*size_batch

        x_data = []
        y_data = []
        for i in range(batch_begin, batch_end):
            #x_data.append(util.convert_to_one_hot(words[i], words2int, vocab_size)) #caso one-hot
            x_data.append(util.get_index(words[i], words2int))  #caso indice per parola
            y_data.append(labels[i])

        x_tensor = Variable(torch.FloatTensor(x_data))
        y_tensor = Variable(torch.FloatTensor(y_data))

        optimizer.zero_grad()

        y_prob, (hn, cn) = model(torch.unsqueeze(x_tensor,1), (h0, c0))
        loss = criterion(torch.squeeze(y_prob,2), torch.unsqueeze(y_tensor,1))
        loss = torch.mul(class_weights, loss )               
        loss= torch.mean(loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("Average Loss: {}".format(epoch_loss/num_batches))

    
    print(" Validation: ")
    epoch_acc = 0
    num_batches_val = len(val) // size_batch
    print("Number of batches for validation: {}".format(num_batches_val))
    for batch in range(num_batches_val):
        batch_begin = batch*size_batch
        batch_end = (batch+1)*size_batch

        x_data_val = []
        y_data_val = []
        for i in range(batch_begin, batch_end):
            #x_data_val.append(util.convert_to_one_hot(words[i], words2int, vocab_size))
            x_data_val.append(util.get_index(words[i], words2int))  #caso indice per parola
            y_data_val.append(labels[i])

    x_tensor_val = Variable(torch.FloatTensor(x_data))
    y_tensor_val = Variable(torch.FloatTensor(y_data))


    y_pred_val,_ = model(torch.unsqueeze(x_tensor_val,1), (hn ,cn))
    #print(y_pred_val.shape)
    output = torch.max(y_pred_val, dim=1)
    prediction = []
    for value in output.indices.data.numpy():
        if (value == 0): prediction.append(0)
        else: prediction.append(1)

    epoch_acc += balanced_accuracy_score(y_data_val, prediction)
    print("Epoch Accuracy: {}".format(epoch_acc))
