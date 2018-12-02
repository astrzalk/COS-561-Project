import numpy as np

class loader(object):
    
    tr_id = [[10,  9,  2, 10,  4,  1,  3,  0,  2,  0,  3,  7,  1,  4,  4,  5,  6,  6, 2,  0],
             [ 7,  6,  5, 11,  9,  2,  4,  3,  0,  1,  8,  8,  7,  7,  3,  2,  7, 10, 6,  5]]
    va_id = [[ 8, 11,  4,  9,  0,],
             [ 6,  5,  8,  8,  0,]]
    te_id = [[ 1,  6, 10,  7,  5,],
             [ 8,  0,  8, 11, 10,]]
    
    def __init__(shuffle, bin_size, batch_size, split):
        """
            Args:
                shuffle (bool):
                bin_size (int):
                batch_size (int):
                split (str): either 'tr', 'va' or 'te'
        """
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        if split == 'tr':
            data_id = tr_id
        elif split =='va':
            data_id = va_id
        else:
            data_id = te_id
        
        self.x = []
        self.y = []
        for i in range(data_id.shape[-1]):
            _x, _y = load_data(data_id[0, i], data_id[1, i])
            self.x.append(_x)
            self.y.append(_y)
        self.x = np.vstack(self.x)
        self.y = np.vstack(self.y)
        
        self.x = self.x[:, -bin_size:]
        
        self.num_batches = self.x.shape[0] // self.batch_size
        self.num_samples = self.num_batches * self.batch_size
        self.step = 0
        
        if self.x.shape[0] % batch_size != 0:
            print("Warning: There are {} extra samples. \
            Take care for test and validation sets".format(
                self.x.shape[0] % batch_size))
        
    def load():
        if self.shuffle:
            if self.step % self.num_batches == 0:
                index  = np.arange(self.x.shape[0])
                np.random.shuffle(index)
                self.x = self.x[index]
                self.y = self.y[index]

        index_start = int((self.step * self.batch_size) % self.num_samples)
        index_end = int((index_start + self.batch_size))
        self.step += 1
        
        return (self.x[index_start:index_end].copy(),
                self.y[index_start:index_end].copy())
