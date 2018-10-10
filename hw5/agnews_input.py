import numpy as np
import csv
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import string

DATADIR = 'ag_news_csv/'

def quantize(data, length=1014):
    alphabet = string.ascii_lowercase
    classes = len(alphabet)
    quantized = []
    for text in data:
        # Alphabet consists of lowercase alphabetical characters
        if len(text) > length:
            text = text[:length]
        elif len(text) < length:
            text = text.ljust(length)

        # really slow
        quantized.append(
            [np.eye(classes, dtype=np.float16)[ord(c) - 97] if c in alphabet else np.zeros(classes, dtype=np.float16) for c in text.lower()])
    return np.array(quantized)

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
        
        self.num_train = int(np.floor((1-.1) * len(self._train_labels)))
        self.index_in_epoch = 0
        self.current_epoch = 0

        self.shuffle_and_split()

    def shuffle_and_split(self):
        labels, data = shuffle(self._train_labels, self._train_data)
        self.train_labels = labels[:self.num_train]
        self.train_data = data[:self.num_train]
        self.validation_labels = labels[self.num_train:]
        self.validation_data = data[self.num_train:]

    def next_batch(self, batch_size, shuffle=True, encoded=True):
        start = self.index_in_epoch

        if self.index_in_epoch + batch_size > len(self.train_labels):
            remaining_data = self.train_data[start:]
            remaining_labels = self.train_labels[start:]
            
            self.current_epoch += 1
            self.shuffle_and_split()
            
            rollover_data = self.train_data[:batch_size-len(remaining)]
            rollover_labels = self.train_labels[:batch_size-len(remaining)]
            
            self.index_in_epoch = len(rollover_labels)
            
            data = remaining_data + rollover_data
            labels = remaining_labels + rollover_labels
            return quantize(data), np.eye(4)[[ord(l)-49 for l in labels]], self.current_epoch

        else:
            self.index_in_epoch += batch_size
            data = self.train_data[start:self.index_in_epoch]
            labels = self.train_labels[start:self.index_in_epoch]
            return quantize(data), np.eye(4)[[ord(l)-49 for l in labels]], self.current_epoch

    def get_validation_set(self):
        return quantize(self.validation_data), np.eye(4)[[ord(l)-49 for l in self.validation_labels]]


    def get_test_set(self):
        return quantize(self.test_data), np.eye(4)[[ord(l)-49 for l in self.test_labels]]

if __name__ == '__main__':
    agnews = DataSet()
    import time
    while(True):
        agnews.next_batch(128)
        print(time.time())

