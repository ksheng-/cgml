import numpy as np
import csv
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import string

DATADIR = 'ag_news_csv/'
alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\’/\\|_@#$%ˆ&* ̃‘+-=<>()[]{}'
lookup = {c:i for i, c in enumerate(alphabet)}

def quantize(data, length=1014):
    classes = len(alphabet)
    quantized = np.zeros((len(data), length, classes), dtype=np.int8)
    categories = np.eye(classes, dtype=np.int8)
    for i, text in enumerate(data):
        # Alphabet consists of lowercase alphabetical characters
        if len(text) > length:
            text = text[:length]
        elif len(text) < length:
            text = text.ljust(length)
        for j, c in enumerate(text.lower()):
            if c in alphabet:
                quantized[i, j, :] = categories[lookup[c]]
        #  really slow
        #  quantized.append(
            #  [categories[lookup[c]] if c in alphabet else np.zeros(classes, dtype=np.int8) for c in text.lower()])
    return quantized

def one_hot(data, classes=4):
    categories = np.eye(classes, dtype=np.int8)
    return categories[[ord(label)-49 for label in data]]

class DataSet():
    def __init__(self, validation_split=.1):
        self._train_labels = []
        self._train_data = []
        with open(DATADIR + 'train.csv', 'r') as f:
            reader = csv.reader(f)
            for label, title, description in reader:
                self._train_labels.append(label)
                self._train_data.append(title + ' ' + description)

        self.test_labels = []
        self.test_data = []
        with open(DATADIR + 'test.csv', 'r') as f:
            reader = csv.reader(f)
            for label, title, description in reader:
                self.test_labels.append(label)
                self.test_data.append(title + ' ' + description)
        
        self.num_train = int(np.floor((1-validation_split) * len(self._train_labels)))
        self.index_in_epoch = 0
        self.index_in_validation = 0
        self.index_in_test = 0
        self.current_epoch = 0

        self.shuffle_and_split()

    def shuffle_and_split(self):
        labels, data = shuffle(self._train_labels, self._train_data)
        self.train_labels = labels[:self.num_train]
        self.train_data = data[:self.num_train]
        self.validation_labels = labels[self.num_train:]
        self.validation_data = data[self.num_train:]
        self.index_in_validation = 0
        self.index_in_test = 0

    def next_batch(self, batch_size, shuffle=True, encoded=True):
        start = self.index_in_epoch

        if self.index_in_epoch + batch_size > len(self.train_labels):
            remaining_data = self.train_data[start:]
            remaining_labels = self.train_labels[start:]
            
            self.current_epoch += 1
            self.shuffle_and_split()
            
            rollover_data = self.train_data[:batch_size-len(remaining_labels)]
            rollover_labels = self.train_labels[:batch_size-len(remaining_labels)]
            
            self.index_in_epoch = len(rollover_labels)
            
            data = remaining_data + rollover_data
            labels = remaining_labels + rollover_labels
            return quantize(data), one_hot(labels), self.current_epoch

        else:
            self.index_in_epoch += batch_size
            data = self.train_data[start:self.index_in_epoch]
            labels = self.train_labels[start:self.index_in_epoch]
            return quantize(data), one_hot(labels), self.current_epoch
    
    def get_validation_set(self):
        return quantize(self.validation_data), one_hot(self.validation_labels)
   
    # hardcoding small batches because i'm too lazy to reuse the epoch logic
    # and i'm running into GPU memory issues
    def next_10_validation(self):
        start = self.index_in_validation
        self.index_in_validation += 10
        if self.index_in_validation > len(self.validation_labels):
            self.index_in_validation = 0
            return None
        data = self.validation_data[start:self.index_in_validation]
        labels = self.validation_labels[start:self.index_in_validation]
        return quantize(data), one_hot(labels)

    def next_100_test(self):
        start = self.index_in_test
        self.index_in_test += 100
        if self.index_in_test > len(self.test_labels):
            self.index_in_test = 0
            return None
        data = self.test_data[start:self.index_in_test]
        labels = self.test_labels[start:self.index_in_test]
        return quantize(data), one_hot(labels)

if __name__ == '__main__':
    agnews = DataSet()
    import time
    while(True):
        agnews.next_batch(10000)
        print(time.time())

