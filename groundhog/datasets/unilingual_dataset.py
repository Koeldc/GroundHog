import cPickle as pkl
import gzip
import os
import sys
import time

import threading
import Queue
import logging

import numpy

import theano
import theano.tensor as T

import tables

logger = logging.getLogger(__name__)

def load_data(path, valid_path=None, test_path=None, 
              batch_size=128, n=4, n_words=10000,
              n_gram=True, shortlist_dict=None):
    ''' 
    Loads the dataset
    '''

    #############
    # LOAD DATA #
    #############

    print '... initializing data iterators'

    train = PytablesBitextIterator(batch_size, path, n=n, n_words=n_words, use_infinite_loop=False, shortlist_dict=shortlist_dict)
    valid = PytablesBitextIterator(batch_size, valid_path, n=n, n_words=n_words, use_infinite_loop=False, shortlist_dict=shortlist_dict) if valid_path else None
    test = PytablesBitextIterator(batch_size, test_path, n=n, n_words=n_words, use_infinite_loop=False, shortlist_dict=shortlist_dict) if test_path else None

    return train, valid, test


def get_length(path):
    target_table = tables.open_file(path, 'r')
    target_index = target_table.get_node('/indices')

    return target_index.shape[0]


class PytablesBitextFetcher(threading.Thread):
    def __init__(self, parent, start_offset, max_offset=-1, n=4):
        threading.Thread.__init__(self)
        self.parent = parent
        self.start_offset = start_offset
        self.max_offset = max_offset
        self.n = n

    def run(self):

        diter = self.parent

        driver = None
        if diter.can_fit:
            driver = "H5FD_CORE"

        target_table = tables.open_file(diter.target_file, 'r', driver=driver)
        target_data, target_index = (target_table.get_node(diter.table_name),
            target_table.get_node(diter.index_name))

        data_len = target_index.shape[0]

        offset = self.start_offset
        if offset == -1:
            offset = 0
            self.start_offset = offset
            if diter.shuffle:
                offset = np.random.randint(data_len)
        logger.debug("{} entries".format(data_len))
        logger.debug("Starting from the entry {}".format(offset))

        while not diter.exit_flag:
            last_batch = False
            target_ngrams = []

            while len(target_ngrams) < diter.batch_size:
                if offset == data_len or offset == self.max_offset:
                    if diter.use_infinite_loop:
                        offset = self.start_offset
                    else:
                        last_batch = True
                        break

                tlen, tpos = target_index[offset]['length'], target_index[offset]['pos']
                offset += 1
                target_ngrams. append(target_data[tpos:tpos+tlen])
                # for n-grams
                """
                # for each word, grab n-gram
                for tii in xrange(tlen):
                    if tii < self.n+1:
                        ng = numpy.zeros(self.n+1) # 0 </s>
                        ng[self.n-tii:] = target_data[tpos:tpos+tii+1]
                    else:
                        ng = target_data[tpos+tii-(self.n):tpos+tii+1]

                    if diter.shortlist_dict:
                        count = 0
                        for nn in ng:
                            if nn in diter.shortlist_dict:
                                count += 1
                        if count == 0:
                            continue

                    target_ngrams.append(ng)
                   """ 

            if len(target_ngrams):
                diter.queue.put([target_ngrams])
            if last_batch:
                diter.queue.put([None])
                return

class PytablesBitextIterator_UL(object):

    def __init__(self,
                 batch_size,
                 target_file=None,
                 dtype="int64",
                 table_name='/phrases',
                 index_name='/indices',
                 can_fit=False,
                 queue_size=1000,
                 cache_size=1000,
                 shuffle=True,
                 use_infinite_loop=True,
                 n=4,
                 n_words=-1,
                 shortlist_dict=None):

        args = locals()
        args.pop("self")
        self.__dict__.update(args)

        self.exit_flag = False

    def start(self, start_offset=0, max_offset=-1):
        self.queue = Queue.Queue(maxsize=self.queue_size)
        self.gather = PytablesBitextFetcher(self, start_offset, max_offset, n=self.n)
        self.gather.daemon = True
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        batch = self.queue.get()
        if not batch:
            return None
        barray = numpy.array(batch[0])
        X = [x[:-1].astype(self.dtype) for x in barray]
        Y = [y[1:].astype(self.dtype) for y in barray]
        assert len(X[0]) == len(Y[0])
        # get rid of out of vocabulary stuff 
        return X, Y
    
