import util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score
from sklearn.utils import class_weight


from collections import Counter

data_path = "./data/preprocessed_all.txt"

train_dim   = 0.8
val_dim     = 0.1
test_dim    = 0.1 
num_epochs  = 20
size_batch  = 100
hidden1_dim = 400
hidden2_dim = 100


# Edited from feedforward_neural_network_sparse_input.py
class Perceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        y = F.log_softmax(x, dim=1)
        return y

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
    num_classes = len(set(labels))
    print("Vocabulary Size:     {}".format(vocab_size))
    print("Number of Classes:   {}".format(num_classes))

    return train, val, test, words, labels, vocabulary, vocab_size, num_classes, words2int, int2words

print("Loading Data...")
train, val, test, words, labels, vocabulary, vocab_size, num_classes, words2int, int2words = load_data(data_path)

model       = Perceptron(input_dim=vocab_size, output_dim=num_classes)
criterion   = nn.CrossEntropyLoss()
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
    epoch_balanced = epoch_loss = epoch_prec = 0
    for batch in range(num_batches):
        # if((batch+1) % 50 == 0 or (batch+1) == num_batches): 
        #     print("Training on batch: {}/{}".format(batch+1,num_batches))
        batch_begin = batch*size_batch
        batch_end = (batch+1)*size_batch

        x_data = []
        y_data = []
        for i in range(batch_begin, batch_end):
            x_data.append(util.convert_to_one_hot(words[i], words2int, vocab_size))
            y_data.append(labels[i])

        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.int64)

        optimizer.zero_grad()

        y_pred = model(x_tensor)

        loss = criterion(y_pred, y_tensor)
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
            x_data_val.append(util.convert_to_one_hot(words[i], words2int, vocab_size))
            y_data_val.append(labels[i])

    x_tensor_val = torch.tensor(x_data, dtype=torch.float32)
    y_tensor_val = torch.tensor(y_data, dtype=torch.int64)

    y_pred_val = model(x_tensor_val)
    #print(y_pred_val.shape)
    output = torch.max(y_pred_val, dim=1)
  
    prediction = []
    for value in output[1].data.numpy():
        if (value == 0): prediction.append(0)
        else: prediction.append(1)

    epoch_acc += accuracy_score(y_data_val, prediction)
    epoch_balanced  += balanced_accuracy_score(y_data_val, prediction)
    epoch_prec = precision_score(y_data_val, prediction)
    print("Epoch balanced Accuracy: {}".format(epoch_balanced))
    print("Epoch  Accuracy: {}".format(epoch_acc))
    print("Epoch  Precision: {}".format(epoch_prec))