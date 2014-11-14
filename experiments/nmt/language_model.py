import numpy
import logging
import pprint
import operator
import itertools

import ipdb

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.datasets import PytablesBitextIterator_UL
from experiments.nmt import prototype_lm_state

from groundhog.mainLoop import MainLoop

from groundhog.layers import\
        Layer,\
        MultiLayer,\
        SoftmaxLayer,\
        HierarchicalSoftmaxLayer,\
        LSTMLayer, \
        RecurrentLayer,\
        RecurrentMultiLayer, \
        RecurrentMultiLayerInp, \
        RecurrentMultiLayerShortPath, \
        RecurrentMultiLayerShortPathInp, \
        RecurrentMultiLayerShortPathInpAll, \
        RecursiveConvolutionalLayer,\
        UnaryOp,\
        Shift,\
        LastState,\
        DropOp,\
        Concatenate

def create_padded_batch(state, x, y, return_dict=False):
    """A callback given to the iterator to transform data in suitable format

    :type x: list
    :param x: list of numpy.array's, each array is a batch of phrases
        in some of source languages

    :type y: list
    :param y: same as x but for target languages

    :param new_format: a wrapper to be applied on top of returned value

    :returns: a tuple (X, Xmask, Y, Ymask) where
        - X is a matrix, each column contains a source sequence
        - Xmask is 0-1 matrix, each column marks the sequence positions in X
        - Y and Ymask are matrices of the same format for target sequences
        OR new_format applied to the tuple

    Notes:
    * actually works only with x[0] and y[0]
    * len(x[0]) thus is just the minibatch size
    * len(x[0][idx]) is the size of sequence idx
    """

    mx = state['seqlen']
    my = state['seqlen']
    if state['trim_batches']:
        # Similar length for all source sequences
        mx = numpy.minimum(state['seqlen'], max([len(xx) for xx in x[0]]))+1
        # Similar length for all target sequences
        my = numpy.minimum(state['seqlen'], max([len(xx) for xx in y[0]]))+1

    # Batch size
    n = x[0].shape[0]

    X = numpy.zeros((mx, n), dtype='int64')
    Y = numpy.zeros((my, n), dtype='int64')
    Xmask = numpy.zeros((mx, n), dtype='float32')
    Ymask = numpy.zeros((my, n), dtype='float32')

    # Fill X and Xmask
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        if mx < len(x[0][idx]):
            X[:mx, idx] = x[0][idx][:mx]
        else:
            X[:len(x[0][idx]), idx] = x[0][idx][:mx]

        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            X[len(x[0][idx]):, idx] = state['null_sym']

        # Initialize Xmask column with ones in all positions that
        # were just set in X
        Xmask[:len(x[0][idx]), idx] = 1.
        if len(x[0][idx]) < mx:
            Xmask[len(x[0][idx]), idx] = 1.

    # Fill Y and Ymask in the same way as X and Xmask in the previous loop
    for idx in xrange(len(y[0])):
        Y[:len(y[0][idx]), idx] = y[0][idx][:my]
        if len(y[0][idx]) < my:
            Y[len(y[0][idx]):, idx] = state['null_sym']
        Ymask[:len(y[0][idx]), idx] = 1.
        if len(y[0][idx]) < my:
            Ymask[len(y[0][idx]), idx] = 1.

    null_inputs = numpy.zeros(X.shape[1])

    # We say that an input pair is valid if both:
    # - either source sequence or target sequence is non-empty
    # - source sequence and target sequence have null_sym ending
    # Why did not we filter them earlier?
    for idx in xrange(X.shape[1]):
        if numpy.sum(Xmask[:,idx]) == 0 and numpy.sum(Ymask[:,idx]) == 0:
            null_inputs[idx] = 1
        if Xmask[-1,idx] and X[-1,idx] != state['null_sym']:
            null_inputs[idx] = 1
        if Ymask[-1,idx] and Y[-1,idx] != state['null_sym']:
            null_inputs[idx] = 1

    valid_inputs = 1. - null_inputs

    # Leave only valid inputs
    X = X[:,valid_inputs.nonzero()[0]]
    Y = Y[:,valid_inputs.nonzero()[0]]
    Xmask = Xmask[:,valid_inputs.nonzero()[0]]
    Ymask = Ymask[:,valid_inputs.nonzero()[0]]
    if len(valid_inputs.nonzero()[0]) <= 0:
        return None

    # Unknown words
    X[X >= state['n_sym']] = state['unk_sym']
    Y[Y >= state['n_sym']] = state['unk_sym']

    if return_dict:
        return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask

def get_batch_iterator(state):

    class Iterator(PytablesBitextIterator_UL):

        def __init__(self, *args, **kwargs):
            PytablesBitextIterator_UL.__init__(self, *args, **kwargs)
            self.batch_iter = None
            self.peeked_batch = None

        def get_homogenous_batch_iter(self):
            while True:
                k_batches = state['sort_k_batches']
                batch_size = state['bs']
                data = [PytablesBitextIterator_UL.next(self) for k in range(k_batches)]
                x = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(0), data))))
                y = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(1), data))))
                lens = numpy.asarray([map(len, x), map(len, y)])
                order = numpy.argsort(lens.max(axis=0)) if state['sort_k_batches'] > 1 \
                        else numpy.arange(len(x))
                for k in range(k_batches):
                    indices = order[k * batch_size:(k + 1) * batch_size]
                    batch = create_padded_batch(state, [x[indices]], [y[indices]],
                            return_dict=True)
                    if batch:
                        yield batch

        def next(self, peek=False):
            if not self.batch_iter:
                self.batch_iter = self.get_homogenous_batch_iter()

            if self.peeked_batch:
                # Only allow to peek one batch
                assert not peek
                logger.debug("Use peeked batch")
                batch = self.peeked_batch
                self.peeked_batch = None
                return batch

            if not self.batch_iter:
                raise StopIteration
            batch = next(self.batch_iter)
            if peek:
                self.peeked_batch = batch
            return batch

    data = Iterator(
        batch_size=int(state['bs']),
        target_file=state['target'],
        can_fit=False,
        queue_size=1000,
        shuffle=state['shuffle'],
        use_infinite_loop=state['use_infinite_loop'],
        n_words=state['n_sym']
        )
    return data

class LM_builder(object):

    def __init__(self, state, rng):
        self.state = state
        self.rng = rng
        self.skip_init = True if self.state['reload'] else False

        self.__create_layers__()

    def __create_layers__(self):
        
        self.emb_words = MultiLayer(
            rng,
            n_in=self.state['n_sym'],
            n_hids=self.state['rank_n_approx'],
            activation=eval(self.state['rank_n_activ']),
            init_fn=state['weight_init_fn'] if not self.skip_init else "sample_zeros",
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'],
            learn_bias = True,
            bias_scale=self.state['bias'],
            name='lm_emb_words')

        self.rec = eval(self.state['rec_layer'])(
                rng,
                n_hids=self.state['dim'],
                activation = eval(self.state['activ']),
                bias_scale = self.state['bias'],
                scale=self.state['rec_weight_scale'],
                init_fn=self.state['rec_weight_init_fn']
                    if not self.skip_init
                    else "sample_zeros",
                weight_noise=self.state['weight_noise_rec'],
                #gating=self.state['rec_gating'],
                #gater_activation=self.state['rec_gater'],
                #reseting=self.state['rec_reseting'],
                #reseter_activation=self.state['rec_reseter'],
                 name='lm_rec')

        self.output_layer = SoftmaxLayer(
            rng,
            self.state['dim'],
            self.state['n_sym'],
            self.state['out_scale'],
            self.state['out_sparse'],
            init_fn="sample_weights_classic",
            weight_noise=self.state['weight_noise'],
            sum_over_time=True,
            name='lm_out')

        """
        # this is currently not used
        self.shortcut = MultiLayer(
            rng,
            n_in=self.state['n_in'],
            n_hids=eval(self.state['inpout_nhids']),
            activations=eval(self.state['inpout_activ']),
            init_fn='sample_weights_classic',
            weight_noise = self.state['weight_noise'],
            scale=eval(self.state['inpout_scale']),
            sparsity=eval(self.state['inpout_sparse']),
            learn_bias=eval(self.state['inpout_learn_bias']),
            bias_scale=eval(self.state['inpout_bias']),
            name='lm_shortcut')
        """

    def build(self):
        """
        Build Computational Graph
        """
        self.x = TT.lmatrix('x')
        self.x_mask = TT.matrix('x_mask')
        self.y = TT.lmatrix('y')
        self.y_mask = TT.matrix('y_mask')

        self.inputs = [self.x, self.y, self.x_mask, self.y_mask]

        self.x_emb = self.emb_words(self.x, no_noise_bias=self.state['no_noise_bias'])

        self.h0 = theano.shared(numpy.zeros(self.state['dim'], dtype='float32'))

        self.reset = TT.scalar('reset')

        self.rec_layer = self.rec(self.x_emb, self.x_mask, 
                    init_state=self.h0*self.reset,
                    no_noise_bias=self.state['no_noise_bias'],
                    truncate_gradient=self.state['cutoff'],
                    batch_size=self.x_emb.shape[1],
                    #nsteps=self.x_emb.shape[0]
                    )
 
        self.train_model = self.output_layer(self.rec_layer,
        no_noise_bias=self.state['no_noise_bias']).train(target=self.y,
                mask=self.y_mask)

        # TODO should double check this..
        self.nw_h0 = self.rec_layer.out[self.rec_layer.out.shape[0]-1]

        if state['carry_h0']:
            train_model.updates += [(self.h0, self.nw_h0)]

if __name__ == '__main__':
    state = prototype_lm_state()

    rng = numpy.random.RandomState(state['seed'])

    train_data = get_batch_iterator(state) 
    train_data.start()
    text= train_data.next()
    model = LM_builder(state, rng)
    model.build()

    import ipdb; ipdb.set_trace()

    lm_model = LM_Model(
        cost_layer = model.train_model,
        weight_noise_amount=state['weight_noise_amount'],
        valid_fn = valid_fn,
        clean_before_noise_fn = False,
        noise_fn = None,
        rng = rng)
 
    algo = eval(state['algo'])(lm_model, state, train_data)

    main = MainLoop(train_data, None, None, lm_model, algo, state, None,
            reset=state['reset'],
            hooks=[RandomSamplePrinter(state, lm_model, train_data)]
                if state['hookFreq'] >= 0 
                else None)

    if state['reload']:
        main.load()
    if state['loopIters'] > 0:
        main.main()


