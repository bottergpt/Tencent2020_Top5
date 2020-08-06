from keras.optimizers import Optimizer
import keras.backend as K
import keras
from collections import OrderedDict


def get_seq_input_layers(cols, seq_length):
    print("Prepare input layer:", cols)
    inputs_dict = OrderedDict()
    for col in cols:
        inputs_dict[col] = keras.Input(shape=(seq_length, ),
                                       dtype="int32",
                                       name=col)
    return inputs_dict


def get_emb_layer(emb_matrix, seq_length=None, trainable=False):
    embedding_dim = emb_matrix.shape[-1]
    input_dim = emb_matrix.shape[0]
    emb_layer = keras.layers.Embedding(input_dim,
                                       embedding_dim,
                                       input_length=seq_length,
                                       weights=[emb_matrix],
                                       dtype="float32",
                                       trainable=trainable)
    return emb_layer


def get_callbacks(count):

    # ckptpath = os.path.join(CKPT_BASE_DIR,
    #                         f'CKPT_{TRAIN_MARKER}_FOLD_{count}.ckpt')
    # checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     ckptpath,
    #     monitor='val_acc',
    #     verbose=1,
    #     save_best_only=True,
    #     mode='max',
    #     save_weights_only=True)

    earlystop_callback = keras.callbacks.EarlyStopping(
        monitor="val_acc",
        min_delta=0.00001,
        patience=3,
        verbose=1,
        mode="max",
        baseline=None,
        restore_best_weights=True,
    )
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                           factor=0.25,
                                                           patience=1,
                                                           min_delta=2e-4,
                                                           min_lr=2e-5)
    callbacks = [earlystop_callback,
                 reduce_lr_callback]  # , checkpoint_callback] # 不存ckpt
    return callbacks


class AdamW(Optimizer):
    def __init__(
            self,
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            weight_decay=1e-4,  # decoupled weight decay (1/4)
            epsilon=1e-8,
            decay=0.,
            **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            # decoupled weight decay (2/4)
            self.wd = K.variable(weight_decay, name='weight_decay')
        self.epsilon = epsilon
        self.initial_decay = decay


#     @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd  # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. /
                   (1. +
                    self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            # decoupled weight decay (4/4)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.wd)),
            'epsilon': self.epsilon
        }
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))