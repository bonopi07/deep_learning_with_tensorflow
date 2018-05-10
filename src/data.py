import numpy as np
import _pickle
import random
import queue as q


class DataLoader:
    def __init__(self, mal_paths, ben_paths, batch_size, mode):
        self.class_description = 'PE File Data Loader'
        self.iter_mode = mode  # for mini-batch data feeding

        '''
            initialize member variable
        '''
        self.mal_paths = mal_paths
        self.ben_paths = ben_paths
        self.mal_data = q.Queue()
        self.ben_data = q.Queue()

        self.number_of_data = len(self.mal_paths) + len(self.ben_paths)

        '''
            set label data
        '''
        self.mal_label = [[0, 1] for _ in range(len(self.mal_paths))]
        self.ben_label = [[1, 0] for _ in range(len(self.ben_paths))]

        '''
            set batch size
        '''
        self.batch_size = batch_size

        '''
            allocate all data into memory
        '''
        for path in self.mal_paths:
            self.mal_data.put(_pickle.load(open(path, 'rb')))
        for path in self.ben_paths:
            self.ben_data.put(_pickle.load(open(path, 'rb')))
        self.mal_data = list(self.mal_data.queue)
        self.ben_data = list(self.ben_data.queue)

        pass

    def get_batch(self):
        return self.batch_size

    def __len__(self):
        return self.number_of_data

    def __iter__(self):
        half_batch_size = int(self.batch_size // 2)

        if self.iter_mode == 'train':  # mini-batch
            while True:
                '''
                    initialize batch data/label
                '''
                batch_data_lists, batch_label_lists = list(), list()

                '''
                    shuffle index list
                '''
                mal_idx_lists = np.arange(len(self.mal_data))
                ben_idx_lists = np.arange(len(self.ben_data))

                random.shuffle(mal_idx_lists)
                random.shuffle(ben_idx_lists)

                '''
                    create batch file/label list
                '''
                for mal_idx, ben_idx in zip(mal_idx_lists[:half_batch_size], ben_idx_lists[:half_batch_size]):
                    # batch malware data
                    batch_data_lists.append(self.mal_data[mal_idx])
                    batch_label_lists.append(self.mal_label[mal_idx])
                    # batch benignware data
                    batch_data_lists.append(self.ben_data[ben_idx])
                    batch_label_lists.append(self.ben_label[ben_idx])

                yield (batch_data_lists, batch_label_lists)
        else:  # evaluation mode
            for data, label in zip(np.concatenate((self.mal_data, self.ben_data), axis=0),
                                   np.concatenate((self.mal_label, self.ben_label), axis=0)):
                yield [data], [label]
