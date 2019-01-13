import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Lambda, Input, concatenate, Flatten
from keras.applications.resnet50 import ResNet50
from misc import Option, wrapped_partial
opt = Option('./config.json')


def get_model(target_model, img_size, num_classes):
    if target_model == 'triplet':
        model = Triplet().get_model(img_size, num_classes)
    else:
        raise Exception('Unknown model: {}'.format(target_model))
    return model


class Triplet:
    def __init__(self):
        self.alpha = opt.alpha
        self.embd_dim = opt.embd_dim

    def triplet_loss(self, y_true, y_pred, embd_dim=128, alpha=0.2):
        anchor = y_pred[:,0:embd_dim]
        positive = y_pred[:,embd_dim:embd_dim*2]
        negative = y_pred[:,embd_dim*2:embd_dim*3]

        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor - positive), axis=1)

        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor - negative), axis=1)

        # compute loss
        basic_loss = pos_dist - neg_dist + alpha
        loss = K.maximum(basic_loss, 0.0)
        return loss

    def count_nonzero(self, y_true, y_pred):
        """
        Custom metric
        Returns count of nonzero embeddings
        """
        return(tf.count_nonzero(y_pred))

    def check_nonzero(self, y_true, y_pred):
        """
        Custom metric
        Returns sum of all embeddings
        """
        return(K.sum(y_pred))

    def get_model(self, img_size, num_classes):
        shape = (img_size, img_size, 3)
        a = Input(shape=shape)
        p = Input(shape=shape)
        n = Input(shape=shape)

        resnet = ResNet50(include_top=False, weights=None, input_shape=shape, pooling='avg')
        net = resnet.output
        net = Dense(self.embd_dim, activation='linear')(net)
        net = Lambda(lambda x: K.l2_normalize(x, axis=-1))(net)
        base_model = Model(resnet.input, net, name='resnet50')

        a_emb = base_model(a)
        p_emb = base_model(p)
        n_emb = base_model(n)

        merge_vec = concatenate([a_emb, p_emb, n_emb], axis=-1)

        model = Model(inputs=[a, p, n], outputs=merge_vec)
        optm = keras.optimizers.Nadam(opt.lr)
        model.compile(loss=wrapped_partial(self.triplet_loss,
                                           embd_dim=self.embd_dim,
                                           alpha=self.alpha),
                      optimizer=optm,
                      metrics=['accuracy', self.count_nonzero, self.check_nonzero])
        model.summary()
        return model
