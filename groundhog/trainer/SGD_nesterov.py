import numpy
import time

import theano
import theano.tensor as TT
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils.utils import print_time, print_mem, const


class SGD(object):
    def __init__(self,
                 model,
                 state,
                 data,
                 rmsprop_decay=None,
                 eps=1e-5):
        """
        This class implements the Nesterov Momentum in the formulation at the below paper:
            http://arxiv.org/pdf/1212.0901v2.pdf
        With rmsprop as well.
        Parameters:
            :param model:
                Class describing the model used. It should provide the
                 computational graph to evaluate the model, and have a
                 similar structure to classes on the models folder
            :param state:
                Dictionary containing the current state of your job. This
                includes configuration of the job, specifically the seed,
                the startign damping factor, batch size, etc. See main.py
                for details
            :param data:
                Class describing the dataset used by the model
            :param rmsprop_decay:
                Decay parameter for rmsprop.
        """

        #####################################
        # Step 0. Constructs shared variables
        #####################################
        bs = state['bs']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])
        srng = RandomStreams(self.rng.randint(213))
        self.rmsprop_decay = rmsprop_decay
        self.gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name)
                   for p in model.params]
        self.rms_gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name="rms_%s" % p.name)
                   for p in model.params]
        self.mean_gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name="meangs_%s" % p.name)
                   for p in model.params]

        self.eps = eps
        #self.mgs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
        #                                     dtype=theano.config.floatX),
        #                        name=p.name)
        #           for p in model.params]
        self.step = 0
        self.bs = bs
        self.state = state
        self.data = data
        self.step_timer = time.time()
        self.gdata = [theano.shared(numpy.zeros( (2,)*x.ndim,
                                                dtype=x.dtype),
                                    name=x.name) for x in model.inputs]

	if 'profile' not in self.state:
            self.state['profile'] = 0

        ###################################
        # Step 1. Compile training function
        ###################################
        print 'Constructing grad function'
        loc_data = self.gdata
        self.prop_exprs = [x[1] for x in model.properties]
        self.prop_names = [x[0] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]
        rval = theano.clone(model.param_grads + self.update_rules + \
                            self.prop_exprs + [model.train_cost],
                            replace=zip(model.inputs, loc_data))
        nparams = len(model.params)
        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]

        # Clip the momentum-applied gradient
        moment_gs = []
        rms_gs = []
	mean_gs = []
        for p, s, g, mg, rms in zip(self.model.params, self.gs, gs, self.mean_gs, self.rms_gs):
            if 'momentum_exclude' in self.model.__dict__ and \
                    p in self.model.momentum_exclude:
                if self.rmsprop_decay is not None:
		    mg_t = (1 - self.rmsprop_decay) * mg + self.rmsprop_decay * g
                    r_t = (1 - self.rmsprop_decay) * g**2 + self.rmsprop_decay * rms
                    r_t = TT.sqrt(r_t - mg**2 + self.eps)
                    rms_gs.append(r_t)
                    mean_gs.append(mg_t)
		    moment_gs.append(g / r_t)
                else:
                    moment_gs.append(g)
            else:
                v_t = s * (state['moment']**2) + (1 + state['moment']) * g
                if self.rmsprop_decay is not None:
		    mg_t = (1 - self.rmsprop_decay) * mg + self.rmsprop_decay * g
                    r_t = (1 - self.rmsprop_decay) * g**2 + self.rmsprop_decay * rms
                    r_t = TT.sqrt(r_t - mg**2 + self.eps)
                    rms_gs.append(r_t)
                    mean_gs.append(mg_t)
		    moment_gs.append(g / r_t)
                else:
                    moment_gs.append(v_t)

        norm_gs = TT.sqrt(sum(TT.sum(x**2)
            for x,p in zip(moment_gs,
                           self.model.params) if p not in self.model.myparams))

        if 'cutoff' in state and state['cutoff'] > 0:
            c = numpy.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_gs))
            _gs = []

            for g,p in zip(moment_gs, self.model.params):
                if p not in self.model.myparams:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       TT.switch(notfinite, numpy.float32(.1)*p,
                           tmpg))
                else:
                    _gs.append(g)
            gs = _gs

        store_gs = [(s, g) for s, g in zip(self.gs, gs)]
        store_rms = [(rms_gi, rms_g) for rms_gi, rms_g in zip(self.rms_gs, rms_gs)]
	store_mgs = zip(self.mean_gs, mean_gs)
        updates = []
        if self.rmsprop_decay is not None:
            updates += store_rms
	    updates += store_mgs

        updates += store_gs + [(s[0], r) for s,r in zip(model.updates, rules)]

        print 'Compiling grad function'
        st = time.time()
        self.train_fn = theano.function([], outs,
                                        name='train_function',
                                        updates = updates,
                                        givens = zip(model.inputs, loc_data),
                                        profile=self.state['profile'])

        print 'took', time.time() - st

        lr = TT.scalar('lr')
        self.lr = numpy.float32(state['lr'])
        new_params = [p - lr * g for p, g in zip(model.params, self.gs)]

        self.update_fn = theano.function([lr], [], name='update_function',
                                         allow_input_downcast=True,
                                         updates = zip(model.params, new_params),
                                         profile=self.state['profile'])

        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + \
                       ['cost',
                        'error',
                        'time_step',
                        'whole_time',
                        'lr']

    def __call__(self):
        batch = self.data.next()
        # Perturb the data (! and the model)
        if isinstance(batch, dict):
            batch = self.model.perturb(**batch)
        else:
            batch = self.model.perturb(*batch)
        # Load the dataset into GPU
        # Note: not the most efficient approach in general, as it involves
        # each batch is copied individually on gpu
        if isinstance(batch, dict):
            for gdata in self.gdata:
                gdata.set_value(batch[gdata.name], borrow=True)
        else:
            for gdata, data in zip(self.gdata, batch):
                gdata.set_value(data, borrow=True)
        # Run the trianing function
        g_st = time.time()
        rvals = self.train_fn()

        for schedule in self.schedules:
            schedule(self, rvals[-1])

        self.update_fn(self.lr)
        g_ed = time.time()
        self.state['lr'] = float(self.lr)
        cost = rvals[-1]
        # if numpy.isnan(cost) or numpy.isinf(cost):
        #    raise Exception('Got NaN in the cost!')
        self.old_cost = cost
        whole_time = time.time() - self.step_timer
        if self.step % self.state['trainFreq'] == 0:
            msg = '.. iter %4d cost %.3f'
            vals = [self.step, cost]
            for dx, prop in enumerate(self.prop_names):
                msg += ' '+prop+' %.2e'
                vals += [float(numpy.array(rvals[dx]))]
            msg += ' step time %s whole time %s lr %.2e'
            vals += [print_time(g_ed - g_st),
                     print_time(time.time() - self.step_timer),
                     float(self.lr)]
            print msg % tuple(vals)

        self.step += 1

        ret = dict([('cost', float(cost)),
                    ('error', float(cost)),
                    ('lr', float(self.lr)),
                    ('time_step', float(g_ed - g_st)),
                    ('whole_time', float(whole_time))] + zip(self.prop_names, rvals))

        return ret
