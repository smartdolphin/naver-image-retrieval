import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Lambda, Input, concatenate, Flatten
from keras.applications.resnet50 import ResNet50
#from misc import Option
#opt = Option('./config.json')


def get_model(target_model, img_size, num_classes):
    if target_model == 'triplet':
        model = Triplet().get_model(img_size, num_classes)
    else:
        raise Exception('Unknown model: {}'.format(target_model))
    return model


class Triplet:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.embd_dim = 128

    def triplet_loss(self, y_true, y_pred, alpha=0.2):
        anchor = y_pred[:,0:self.embd_dim]
        positive = y_pred[:,self.embd_dim:self.embd_dim*2]
        negative = y_pred[:,self.embd_dim*2:self.embd_dim*3]

        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor - positive), axis=1)

        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor - negative), axis=1)

        # compute loss
        basic_loss = pos_dist - neg_dist + alpha
        loss = K.maximum(basic_loss, 0.0)
        return loss

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
        optm = keras.optimizers.Nadam(1e-3)
        model.compile(loss=self.triplet_loss, optimizer=optm, metrics=None)
        model.summary()
        return model
