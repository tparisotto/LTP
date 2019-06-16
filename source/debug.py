from util import generate_dataset, convert_to_one_hot
import numpy as np

dataset = generate_dataset("../data/preprocessed_all.txt")
words = []
labels = []
for w,l in dataset:
    words.append(w)
    labels.append(l)
vocabulary = sorted(set(words))
words2int = {w: i for i, w in enumerate(vocabulary)}
int2words = {k: w for k, w in enumerate(words2int)}
vocab_size = len(words2int)
y = convert_to_one_hot(words[23],words2int,vocab_size)
print(y)
print(np.sum(y[0]))