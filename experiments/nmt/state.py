"""
    Config file:
        - general inheritance of params heirarchy is 
            Prototype_phrase_state()
                prototype_encdec_state_*
                    prototype_search_state_*

    Therefore, if GHOG is configured with prototype_encdec_state_* for example
    it will have all the state settings in prototype_phrase_state and 
    whatever was added or overwritten in prototype_encdec_state_*

"""

def prototype_phrase_state():
    """This prototype is the configuration used in the paper
    'Learning Phrase Representations using RNN Encoder-Decoder
    for  Statistical Machine Translation' """

    state = {}

    # Random seed
    state['seed'] = 1234
    # Logging level
    state['level'] = 'DEBUG'

    # ----- DATA -----

    # Source sequences
    state['source'] = ["/data/lisatmp3/bahdanau/shuffled/phrase-table.en.h5"]
    # Target sequences
    state['target'] = ["/data/lisatmp3/bahdanau/shuffled/phrase-table.fr.h5"]
    # index -> word dict for the source language
    state['indx_word'] = "/data/lisatmp3/chokyun/mt/ivocab_source.pkl"
    # index -> word dict for the target language
    state['indx_word_target'] = "/data/lisatmp3/chokyun/mt/ivocab_target.pkl"
    # word -> index dict for the source language
    state['word_indx'] = "/data/lisatmp3/chokyun/mt/vocab.en.pkl"
    # word -> index dict for the target language
    state['word_indx_trgt'] = "/data/lisatmp3/bahdanau/vocab.fr.pkl"

    # Source/Target sentence encoding:
    # assumed to be ascii but overwritten if using a unicode language
    state['target_encoding'] = 'ascii'
    state['source_encoding'] = 'ascii'

    # ----- VOCABULARIES -----

    # A string representation for the unknown word placeholder for both language
    state['oov'] = 'UNK'
    # These are unknown word placeholders
    state['unk_sym_source'] = 1
    state['unk_sym_target'] = 1
    # These are end-of-sequence marks
    state['null_sym_source'] = 15000
    state['null_sym_target'] = 15000
    # These are vocabulary sizes for the source and target languages
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    # ----- MODEL STRUCTURE -----

    # The components of the annotations produced by the Encoder
    state['last_forward'] = True
    state['last_backward'] = False
    state['forward'] = False
    state['backward'] = False
    # Turns on "search" mechanism
    state['search'] = False
    # Turns on using the shortcut from the previous word to the current one
    state['bigram'] = True
    # Turns on initialization of the first hidden state from the annotations
    state['bias_code'] = True
    # Turns on using the context to compute the next Decoder state
    state['decoding_inputs'] = True
    # Turns on an intermediate maxout layer in the output
    state['deep_out'] = True
    # Heights of hidden layers' stacks in encoder and decoder
    # WARNING: has not been used for quite while and most probably
    # doesn't work...
    state['encoder_stack'] = 1
    state['decoder_stack'] = 1
    # Use the top-most recurrent layer states as annotations
    # WARNING: makes sense only for hierachical RNN which
    # are in fact currently not supported
    state['take_top'] = True
    # Activates age old bug fix - should always be true
    state['check_first_word'] = True

    state['eps'] = 1e-10

    # ----- MODEL COMPONENTS -----

    # Low-rank approximation activation function
    state['rank_n_activ'] = 'lambda x: x'
    # Hidden-to-hidden activation function
    state['activ'] = 'lambda x: TT.tanh(x)'
    # Nonlinearity for the output
    state['unary_activ'] = 'Maxout(2)'

    # Hidden layer configuration for the forward encoder
    state['enc_rec_layer'] = 'RecurrentLayer'
    state['enc_rec_gating'] = True
    state['enc_rec_reseting'] = True
    state['enc_rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['enc_rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'
    # Hidden layer configuration for the decoder
    state['dec_rec_layer'] = 'RecurrentLayer'
    state['dec_rec_gating'] = True
    state['dec_rec_reseting'] = True
    state['dec_rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['dec_rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'
    # Default hidden layer configuration, which is effectively used for
    # the backward RNN
    # TODO: separate back_enc_ configuration and convert the old states
    # to have it
    state['rec_layer'] = 'RecurrentLayer'
    state['rec_gating'] = True
    state['rec_reseting'] = True
    state['rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'

    # ----- SIZES ----

    # Dimensionality of hidden layers
    state['dim'] = 1000
    # Dimensionality of low-rank approximation (read: word embedding 
    state['rank_n_approx'] = 100
    # k for the maxout stage of output generation
    state['maxout_part'] = 2.

    # ----- BLEU VALIDATION OPTIONS ----

    # Location of the evaluation script
    state['bleu_script'] = None
    # Location of the validation set
    state['validation_set'] = None
    # boolean, whether or not to write the validation set to file    
    state['output_validation_set'] = False
    # Location of the validation set output, if different
    # fom default
    state['validation_set_out'] = None
    # Location of what to compare the output translation to (gt)
    state['validation_set_grndtruth'] = None
    # Beam size during sampling
    state['beam_size'] = None
    # Number of steps between every validation
    state['bleu_val_frequency'] = None


    # ----- WEIGHTS, INITIALIZATION -----

    # This one is bias applied in the recurrent layer. It is likely
    # to be zero as MultiLayer already has bias.
    state['bias'] = 0.

    # Weights initializer for the recurrent net matrices
    state['rec_weight_init_fn'] = 'sample_weights_orth'
    state['rec_weight_scale'] = 1.
    # Weights initializer for other matrices
    state['weight_init_fn'] = 'sample_weights_classic'
    state['weight_scale'] = 0.01

    # ---- REGULARIZATION -----

    # WARNING: dropout is not tested and probably does not work.
    # Dropout in output layer
    state['dropout'] = 1.
    # Dropout in recurrent layers
    state['dropout_rec'] = 1.

    # WARNING: weight noise regularization is not tested
    # and most probably does not work.
    # Random weight noise regularization settings
    state['weight_noise'] = False
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    # A magic gradient clipping option that you should never change...
    state['cutoff_rescale_length'] = 0.

    # ----- TRAINING METHOD -----

    # Turns on noise contrastive estimation instead maximum likelihood
    state['use_nce'] = False

    # Choose optimization algorithm
    state['algo'] = 'SGD_adadelta'

    # Adadelta hyperparameters
    state['adarho'] = 0.95
    state['adaeps'] = 1e-6

    # Early stopping configuration
    # WARNING: was never changed during machine translation experiments,
    # as early stopping was not used.
    state['patience'] = 1
    state['lr'] = 1.
    state['minlr'] = 0

    # Batch size
    state['bs']  = 64
    # We take this many minibatches, merge them,
    # sort the sentences according to their length and create
    # this many new batches with less padding.
    state['sort_k_batches'] = 20

    # Maximum sequence length
    state['seqlen'] = 30
    # Turns on trimming the trailing paddings from batches
    # consisting of short sentences.
    state['trim_batches'] = True
    # Loop through the data
    state['use_infinite_loop'] = True
    # Start from a random entry
    state['shuffle'] = False

    # ----- TRAINING PROCESS -----

    # Prefix for the model, state and timing files
    state['prefix'] = 'phrase_'
    # Specifies whether old model should be reloaded first
    state['reload'] = False
    # When set to 0 each new model dump will be saved in a new file
    state['overwrite'] = 1

    # Number of batches to process
    state['loopIters'] = 3000000
    # Maximum number of minutes to run
    state['timeStop'] = 24*60*7
    # Error level to stop at
    state['minerr'] = -1

    # Reset data iteration every this many epochs
    state['reset'] = -1
    # Frequency of training error reports (in number of batches)
    state['trainFreq'] = 1
    # Frequency of running hooks
    state['hookFreq'] = 13
    # Validation frequency
    state['validFreq'] = 500
    # Model saving frequency (in minutes)
    state['saveFreq'] = 10

    # Sampling hook settings
    state['n_samples'] = 3
    state['n_examples'] = 3

    # Raise exception if nan
    state['on_nan'] = 'raise'
    return state

# zh -> en,  

def prototype_encdec_state_zh_en():
    """
    This is the prototype for translation from zh->en
    """
    state = prototype_phrase_state()
     
    # Source and target sentence
    state['source'] = ["/data/lisatmp3/xukelvin/translation/en-zh/ted/binarized_text.zh.shuf.h5"]
    state['target'] = ["/data/lisatmp3/xukelvin/translation/en-zh/ted/binarized_text.en.shuf.h5"]
    # Word -> Id and Id-> Word Dictionaries
    state['indx_word_target'] = "/data/lisatmp3/xukelvin/translation/en-zh/ted/ivocab.en.pkl"
    state['indx_word'] = "/data/lisatmp3/xukelvin/translation/en-zh/ted/ivocab.zh.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/xukelvin/translation/en-zh/ted/vocab.en.pkl"
    state['word_indx'] = "/data/lisatmp3/xukelvin/translation/en-zh/ted/vocab.zh.pkl"   

    state['null_sym_source'] = 4000
    state['null_sym_target'] = 30000

    state['source_encoding'] = 'utf8'

    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['seqlen'] = 30

    state['dim'] = 1000
    # embedding dimensionality
    state['rank_n_approx'] = 600

    state['bs']  = 80

    return state

def prototype_search_state_zh_en_3000_600():
    """
    This prototype is for zh-> with a large hidden state 
    
    This was used once but it was determined that a smaller
    model overfit a lot 

    Testing with validation set 
    """
    state = prototype_encdec_state_zh_en()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True

    state['dim'] = 3000
    # embedding dimensionality
    state['rank_n_approx'] = 600
    state['seqlen'] = 50

    # validation set for early stopping
    # bleu validation args
    state['bleu_script'] = '/u/xukelvin/Documents/research/machine_trans/multi-bleu.perl'
    state['validation_set_grndtruth'] = '/data/lisatmp3/chokyun/ted/en-zh/clean_IWSLT13.TED.dev2010.en-zh.tok.en'
    state['validation_set'] = '/data/lisatmp3/chokyun/ted/en-zh/clean_IWSLT13.TED.dev2010.en-zh.zh'
    #state['validation_set_out'] = '/data/lisatmp3/xukelvin/translation/zh-en/tedmodels/encdec_30_small_val.txt'
    state['beam_size'] = 10
    state['bleu_val_frequency'] = 5000

    state['prefix'] = '/data/lisatmp3/xukelvin/translation/zh-en/tedmodels/huge/encdec_50_zh_en_3000_600_'
    return state

def prototype_search_state_zh_en_small():
    """
    This prototype is for zh -> english 

    The size is reduced to help prevent overfitting
    """
    state = prototype_encdec_state_zh_en()

    #location of validation set
    state['source'].append("/data/lisatmp3/xukelvin/translation/en-zh/ted/binarized_text.zh.shuf.h5")
    state['target'].append("/data/lisatmp3/xukelvin/translation/en-zh/ted/binarized_text.en.shuf.h5")

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True

    state['dim'] = 800
    # embedding dimensionality
    state['rank_n_approx'] = 300

    # validation set for early stopping
    # bleu validation args
    state['bleu_script'] = '/u/xukelvin/Documents/research/machine_trans/multi-bleu.perl'
    state['validation_set_grndtruth'] = '/data/lisatmp3/chokyun/ted/en-zh/clean_IWSLT13.TED.dev2010.en-zh.tok.en'
    state['validation_set'] = '/data/lisatmp3/chokyun/ted/en-zh/clean_IWSLT13.TED.dev2010.en-zh.zh'
    state['validation_set_out'] = '/data/lisatmp3/xukelvin/translation/zh-en/tedmodels/encdec_30_small_val.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 3000

    state['prefix'] = '/data/lisatmp3/xukelvin/translation/zh-en/tedmodels/encdec_30_zh_en_small_'
    return state

def prototype_search_state_zh_en():
    """
    This prototype is for zh -> english 
    """
    state = prototype_encdec_state_zh_en()

    #location of validation set
    state['source'].append("/data/lisatmp3/xukelvin/translation/en-zh/ted/binarized_text.zh.shuf.h5")
    state['target'].append("/data/lisatmp3/xukelvin/translation/en-zh/ted/binarized_text.en.shuf.h5")

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True

    state['dim'] = 1000
    # embedding dimensionality
    state['rank_n_approx'] = 600

    # validation set for early stopping
    # bleu validation args
    state['bleu_script'] = '/u/xukelvin/Documents/research/machine_trans/multi-bleu.perl'
    state['validation_set_grndtruth'] = '/data/lisatmp3/chokyun/ted/en-zh/clean_IWSLT13.TED.dev2010.en-zh.tok.en'
    state['validation_set'] = '/data/lisatmp3/chokyun/ted/en-zh/clean_IWSLT13.TED.dev2010.en-zh.zh'
    state['validation_set_out'] = '/data/lisatmp3/xukelvin/translation/zh-en/tedmodels/encdec_30_val.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 4000

    state['prefix'] = '/data/lisatmp3/xukelvin/translation/zh-en/tedmodels/encdec_30_zh_en_'

# en -> zh

def prototype_encdec_state_zh():
    """
    This is the prototype for translation from en->zh 
    """
    state = prototype_phrase_state()
     
    # Source and target sentence
    state['target'] = ["/data/lisatmp3/xukelvin/translation/en-zh/ted/binarized_text.zh.shuf.h5"]
    state['source'] = ["/data/lisatmp3/xukelvin/translation/en-zh/ted/binarized_text.en.shuf.h5"]
    # Word -> Id and Id-> Word Dictionaries
    state['indx_word'] = "/data/lisatmp3/xukelvin/translation/en-zh/ted/ivocab.en.pkl"
    state['indx_word_target'] = "/data/lisatmp3/xukelvin/translation/en-zh/ted/ivocab.zh.pkl"
    state['word_indx'] = "/data/lisatmp3/xukelvin/translation/en-zh/ted/vocab.en.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/xukelvin/translation/en-zh/ted/vocab.zh.pkl"   

    state['null_sym_source'] = 30000
    state['null_sym_target'] = 4000

    state['target_encoding'] = 'utf8'

    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['seqlen'] = 30

    state['dim'] = 1000
    # embedding dimensionality
    state['rank_n_approx'] = 600

    state['bs']  = 80

    return state


def prototype_search_state_zh_big_openmt15():
    """
    This is for openmt15
    en-> zh
    """
    state = prototype_encdec_state_zh()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True

    # overwrite this from above
    # Source and target sentence
    state['target'] = ["/data/lisatmp3/xukelvin/translation/en-zh/openmt/binarized_text.zh.shuf.h5"]
    state['source'] = ["/data/lisatmp3/xukelvin/translation/en-zh/openmt/binarized_text.en.shuf.h5"]
    # Word -> Id and Id-> Word Dictionaries
    state['indx_word'] = "/data/lisatmp3/xukelvin/translation/en-zh/openmt/ivocab.en.pkl"
    state['indx_word_target'] = "/data/lisatmp3/xukelvin/translation/en-zh/openmt/ivocab.zh.pkl"
    state['word_indx'] = "/data/lisatmp3/xukelvin/translation/en-zh/openmt/vocab.en.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/xukelvin/translation/en-zh/openmt/vocab.zh.pkl"   

    state['dim'] = 1800
    # embedding dimensionality
    state['rank_n_approx'] = 800

    state['prefix'] = '/data/lisatmp3/xukelvin/translation/en-zh/openmt_models/encdec_30_big_open_mt'
    return state

def prototype_search_state_zh_en_big_openmt15():
    """
    This is for openmt15
    zh -> english
    """
    state = prototype_encdec_state_zh_en()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True

    # overwrite this from above
    # Source and target sentence
    state['source'] = ["/data/lisatmp3/xukelvin/translation/en-zh/openmt/binarized_text.zh.shuf.h5"]
    state['target'] = ["/data/lisatmp3/xukelvin/translation/en-zh/openmt/binarized_text.en.shuf.h5"]
    # Word -> Id and Id-> Word Dictionaries
    state['indx_word_target'] = "/data/lisatmp3/xukelvin/translation/en-zh/openmt/ivocab.en.pkl"
    state['indx_word'] = "/data/lisatmp3/xukelvin/translation/en-zh/openmt/ivocab.zh.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/xukelvin/translation/en-zh/openmt/vocab.en.pkl"
    state['word_indx'] = "/data/lisatmp3/xukelvin/translation/en-zh/openmt/vocab.zh.pkl"   

    state['dim'] = 2000
    # embedding dimensionality
    state['rank_n_approx'] = 600

    state['prefix'] = '/data/lisatmp3/xukelvin/translation/zh-en/openmt_models/encdec_30_big_open_mt_'
    return state

def prototype_search_state_zh_small():
    """
    This prototype integrates the search model into the translation
    """
    state = prototype_encdec_state_zh()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True


    state['dim'] = 1000
    # embedding dimensionality
    state['rank_n_approx'] = 600

    state['prefix'] = '/data/lisatmp3/xukelvin/translation/en-zh/tedmodels/encdec_30_small_'
    return state

def prototype_search_state_zh_huge():
    """
    This prototype integrates the search model into the translation with a very large size
    """
    state = prototype_encdec_state_zh()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True

    state['seqlen'] = 50

    state['dim'] = 2048
    # embedding dimensionality
    state['rank_n_approx'] = 800

    state['prefix'] = '/data/lisatmp3/xukelvin/translation/en-zh/tedmodels/huge/encdec_30_huge_'
    return state


def prototype_search_state_zh_big():
    """
    This prototype integrates the search model into the translation with a larger size
    """
    state = prototype_encdec_state_zh()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True

    state['dim'] = 1500
    # embedding dimensionality
    state['rank_n_approx'] = 800

    state['prefix'] = '/data/lisatmp3/xukelvin/translation/en-zh/tedmodels/encdec_30_big_'
    return state

def prototype_search_state_zh_big_cv():
    """
    This prototype integrates the search model into the translation with a larger size
    
    This model uses cross validation
    """
    state = prototype_encdec_state_zh()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True

    state['dim'] = 1500
    # embedding dimensionality
    state['rank_n_approx'] = 800

    # validation set for early stopping
    # bleu validation args
    state['bleu_script'] = '/u/xukelvin/Documents/research/machine_trans/multi-bleu.perl'
    state['validation_set_grndtruth'] = '/data/lisatmp3/chokyun/ted/en-zh/clean_IWSLT13.TED.dev2010.en-zh.zh'
    state['validation_set'] = '/data/lisatmp3/chokyun/ted/en-zh/clean_IWSLT13.TED.dev2010.en-zh.tok.en'
    state['validation_set_out'] = '/data/lisatmp3/xukelvin/translation/en-zh/tedmodels/encdec_30_big_cv_.txt'
    state['beam_size'] = 30
    state['bleu_val_frequency'] = 4500


    state['prefix'] = '/data/lisatmp3/xukelvin/translation/en-zh/tedmodels/encdec_30_big_cv_'
    return state


# original configs

def prototype_encdec_state():
    """This prototype is the configuration used to train the RNNenc-30 model from the paper
    'Neural Machine Translation by Jointly Learning to Align and Translate' """

    state = prototype_phrase_state()

    # Source and target sentence
    state['target'] = ["/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/binarized_text.shuffled.fr.h5"]
    state['source'] = ["/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/binarized_text.shuffled.en.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word'] = "/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/ivocab.en.pkl"
    state['indx_word_target'] = "/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/ivocab.fr.pkl"
    state['word_indx'] = "/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/vocab.en.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/vocab.fr.pkl"

    state['null_sym_source'] = 30000
    state['null_sym_target'] = 30000

    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['seqlen'] = 30

    state['dim'] = 1000
    # embedding dimensionality
    state['rank_n_approx'] = 620
    state['bs']  = 80

    state['prefix'] = 'encdec_'
    return state


def prototype_search_state():
    """This prototype is the configuration used to train the RNNsearch-50 model from the paper
    'Neural Machine Translation by Jointly Learning to Align and Translate' """

    state = prototype_encdec_state()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix'] = 'search_'
    return state

def prototype_search_state_zh_en_test():
    """
    This prototype is for zh -> english 
    """
    state = prototype_encdec_state_zh_en()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True

    state['dim'] = 1000
    # embedding dimensionality
    state['rank_n_approx'] = 600
    
    # bleu validation args
    state['bleu_script'] = '/u/xukelvin/Documents/research/machine_trans/multi-bleu.perl'
    state['validation_set_grndtruth'] = '/data/lisatmp3/chokyun/ted/en-zh/clean_IWSLT13.TED.dev2010.en-zh.tok.en'
    state['validation_set'] = '/data/lisatmp3/chokyun/ted/en-zh/clean_IWSLT13.TED.dev2010.en-zh.zh'
    state['validation_set_out'] = '/data/lisatmp3/xukelvin/tmp/val_out.txt'
    state['beam_size'] = 1
    state['bleu_val_frequency'] = 1

    state['prefix'] = '/data/lisatmp3/xukelvin/tmp/test_'
    return state


