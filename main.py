# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import h5py
import argparse
import pickle

import itertools
import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from data_loader import triplet_data_loader
from network import get_model
from misc import Option, ModelMGPU
opt = Option('./config.json')


def bind_model(model, embd_net, batch_size):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, db):

        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322

        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        queries = np.asarray(queries)
        query_img = np.asarray(query_img)
        references = np.asarray(references)
        reference_img = np.asarray(reference_img)

        query_img = query_img.astype('float32')
        query_img /= 255
        reference_img = reference_img.astype('float32')
        reference_img /= 255

        get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()],
                                       [embd_net])

        print('inference start')

        # inference
        query_vecs = get_feature_layer([query_img, 0])[0]
        n_epoch =  reference_img.shape[0] // batch_size
        reference_vecs = np.zeros([0, opt.embd_dim])
        offset = 0

        for i in range(n_epoch):
            reference_img_batch = reference_img[offset:offset+batch_size]
            reference_vecs = np.concatenate([reference_vecs, get_feature_layer([reference_img_batch, 0])[0]])
            offset += batch_size

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = zip(references, sim_matrix[i].tolist())
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)

            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# data preprocess
def preprocess(queries, db):
    query_img = []
    reference_img = []
    img_size = (224, 224)

    for img_path in queries:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        query_img.append(img)

    for img_path in db:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        reference_img.append(img)

    return queries, query_img, db, reference_img


def get_sample_generator(ds, batch_size, img_shape, raise_stop_event=False):
    left, limit = 0, ds['a'].shape[0]
    data_inds = []
    for label in set(ds['y'].value):
        y_set = ds['y'][ds['y'].value == label]
        for a_idx, p_idx in itertools.permutations(y_set, 2):
            data_inds.append((a_idx, p_idx))
    np.random.shuffle(data_inds)
    y_inds = [ds['y'][a_idx] for a_idx, _ in data_inds]

    while True:
        right = min(left + batch_size, limit)

        batch_inds = list(np.arange(left, right))
        a = np.zeros((len(batch_inds), img_shape[0], img_shape[1], img_shape[2]))
        p = np.zeros((len(batch_inds), img_shape[0], img_shape[1], img_shape[2]))
        n = np.zeros((len(batch_inds), img_shape[0], img_shape[1], img_shape[2]))

        for i, batch_idx in enumerate(batch_inds):
            a_idx, p_idx = data_inds[batch_idx]
            a[i] = ds['a'][a_idx]
            p[i] = ds['a'][p_idx]

        for i, batch_idx in enumerate(batch_inds):
            _y = y_inds[batch_idx]
            mask = np.array(y_inds) == _y
            negative_idx = np.random.choice(np.where(np.logical_not(mask))[0], size=1)[0]
            n[i] = ds['a'][data_inds[negative_idx][0]]

        X = [a] + [p] + [n]
        Y = np.zeros((len(batch_inds), opt.embd_dim * 3))
        yield X, Y
        left = right
        if right == limit:
            left = 0
            if raise_stop_event:
                raise StopIteration


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=5)
    args.add_argument('--batch_size', type=int, default=32)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epochs
    batch_size = config.batch_size
    num_classes = 1000
    input_shape = (224, 224, 3)  # input image shape

    """ Model """
    model, base_model, embd_net = get_model('triplet', 224, num_classes, opt.base_model)
    bind_model(base_model, embd_net, config.batch_size)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Load data """
        print('dataset path', DATASET_PATH)
        output_path = './data.h5py'
        train_dataset_path = os.path.join(DATASET_PATH, 'train/train_data')

        if nsml.IS_ON_NSML:
            # Caching file
            nsml.cache(triplet_data_loader,
                       data_path=train_dataset_path,
                       img_size=input_shape[:2],
                       output_path=output_path,
                       num_classes=num_classes,
                       train_ratio=1.0)
        else:
            # local에서 실험할경우 dataset의 local-path 를 입력해주세요
            if os.path.exists(output_path) is False:
                triplet_data_loader(train_dataset_path,
                                    input_shape[:2],
                                    output_path=output_path,
                                    num_classes=num_classes,
                                    train_ratio=1.0)

        """ Prepare train/val data  """
        data = h5py.File(output_path, 'r')
        train = data['train']
        dev = data['dev']

        train_size = train['a'].shape[0]
        a_train = train['a'].value.reshape(train_size, 224, 224, 3)
        a_train /= 255

        total_train_samples = 0
        for label in set(train['y'].value):
            y_set = train['y'][train['y'].value == label]
            total_train_samples += len(list(itertools.permutations(y_set, 2)))
        print(train_size, 'train samples > ', total_train_samples)

        train_gen = get_sample_generator({'a': a_train, 'y': train['y']},
                                         batch_size=batch_size,
                                         img_shape=input_shape)
        steps_per_epoch = int(np.ceil(total_train_samples / float(batch_size)))

        dev_size = dev['a'].shape[0]
        a_dev = dev['a'].value.reshape(dev_size, 224, 224, 3)
        a_dev /= 255

        total_dev_samples = 0
        for label in set(dev['y'].value):
            y_set = dev['y'][dev['y'].value == label]
            total_dev_samples += len(list(itertools.permutations(y_set, 2)))
        print(dev_size, 'dev samples > ', total_dev_samples)

        dev_gen = get_sample_generator({'a': a_dev, 'y': dev['y']},
                                       batch_size=batch_size,
                                       img_shape=input_shape)
        validation_steps = int(np.ceil(total_dev_samples / float(batch_size)))

        """ Pre-training data """
        x_train, x_test, y_train, y_test = train_test_split(a_train, train['y'].value,
                                                            test_size=opt.pretrain_test_split,
                                                            random_state=0)
        y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
        if x_test is not []:
            y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

        """ Callback """
        monitor = 'acc'
        callbacks = [ReduceLROnPlateau(monitor=monitor, patience=3)]


        """ Pre-training base model first """
        if base_model is not None:
            optm = keras.optimizers.Nadam(opt.pretrain_lr)
            net = keras.layers.Dense(num_classes, activation='softmax')(embd_net)
            pretrain = keras.models.Model(inputs=[base_model.input], outputs=net)
            if opt.num_gpus > 1:
                pretrain = ModelMGPU(pretrain, gpus=opt.num_gpus)
            pretrain.compile(loss='categorical_crossentropy',
                             optimizer=optm,
                             metrics=['accuracy'])
            pretrain.summary()
            train_datagen = ImageDataGenerator(rotation_range=40,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True,
                                               fill_mode='nearest')
            train_generator = train_datagen.flow(x_train, y_train, batch_size=opt.pretrain_batch_size)
            pretrain_steps_per_epoch = int(np.ceil(x_train.shape[0] / float(opt.pretrain_batch_size)))
            if x_test.size == 0:
                test_datagen = ImageDataGenerator()
                test_generator = test_datagen.flow(x_test, y_test, batch_size=opt.pretrain_batch_size)
                test_validation_steps = int(np.ceil(x_test.shape[0] / float(opt.pretrain_batch_size)))

            res = pretrain.fit_generator(train_generator,
                                         epochs=opt.pretrain_n_epoch,
                                         steps_per_epoch=pretrain_steps_per_epoch,
                                         validation_data=test_generator if x_test.size == 0 else None,
                                         validation_steps=test_validation_steps if x_test.size == 0 else None,
                                         shuffle=True,
                                         workers=4,
                                         callbacks=callbacks)

        """ Training loop """
        for epoch in range(nb_epoch):
            res = model.fit_generator(generator=train_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=1,
                                      validation_data=dev_gen if dev_size > 0 else None,
                                      validation_steps=validation_steps if dev_size > 0 else None,
                                      shuffle=True,
                                      callbacks=callbacks)
            print(res.history)
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
            nsml.save(epoch)
        data.close()
