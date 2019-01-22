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
import tqdm

from nsml import DATASET_PATH
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from data_loader import train_data_loader
from network import get_model
from util import lr_schedule
from misc import Option, ModelMGPU, ThreadsafeIter
opt = Option('./config.json')


def bind_model(model, batch_size):
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
                                       [model.layers[-1].output])

        print('inference start: ', opt.infer_dist)

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

        if opt.infer_dist == 'l2':
            # l2 distance
            sim_matrix = []
            for query_vec in query_vecs:
                dist = np.sum(np.absolute(query_vec - reference_vecs), axis=1)
                sim_matrix.append(dist)
            sim_matrix = np.array(sim_matrix)
        else:
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


def get_sample_generator(ds, batch_size, img_shape, model=None, hard=False, raise_stop_event=False):
    left, limit = 0, ds['a'].shape[0]
    data_inds = []
    for label in set(ds['y']):
        y_set = ds['y'][ds['y'] == label]
        for a_idx, p_idx in itertools.permutations(y_set, 2):
            data_inds.append((a_idx, p_idx))
    np.random.shuffle(data_inds)
    y_inds = [ds['y'][a_idx] for a_idx, _ in data_inds]
    a_inds = [a_idx for a_idx, _ in data_inds]
    n_epoch =  len(data_inds) // batch_size
    if len(data_inds) % batch_size > 0:
        n_epoch += 1
    is_first_step = True

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

        if hard and is_first_step:
            is_first_step = False
            with tf.device('/cpu:0'):
                emb_mat = np.zeros([0, opt.embd_dim])
                offset = 0
                for i in tqdm.trange(n_epoch):
                    tmp = ds['a'][a_inds][offset:offset+batch_size]
                    emb_mat = np.concatenate([emb_mat, model([tmp, 0])[0]])
                    offset += batch_size
                scores = emb_mat @ emb_mat.T

        for i, batch_idx in enumerate(batch_inds):
            _y = y_inds[batch_idx]
            mask = np.array(y_inds) == _y
            if hard:
                negative_idx = np.ma.array(scores[a_inds[batch_idx]], mask=mask).argmax()
            else:
                negative_idx = np.random.choice(np.where(np.logical_not(mask))[0], size=1)[0]
            n[i] = ds['a'][data_inds[negative_idx][0]]

        X = [a] + [p] + [n]
        Y = np.zeros((len(batch_inds), opt.embd_dim * 3))
        yield X, Y
        left = right
        if right == limit:
            left = 0
            is_first_step = True
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
    model, base_model = get_model('triplet', 224, num_classes, opt.base_model)
    bind_model(base_model, config.batch_size)
    get_feature_layer = K.function([base_model.layers[0].input] + [K.learning_phase()],
                                   [base_model.layers[-1].output])

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Load data """
        print('dataset path', DATASET_PATH)
        output_path = ['./img_list.pkl', './label_list.pkl']
        train_dataset_path = os.path.join(DATASET_PATH, 'train/train_data')

        if nsml.IS_ON_NSML:
            # Caching file
            nsml.cache(train_data_loader,
                       data_path=train_dataset_path,
                       img_size=input_shape[:2],
                       output_path=output_path)
        else:
            if not os.path.exists(output_path[0]) or \
               not os.path.exists(output_path[1]):
                # local에서 실험할경우 dataset의 local-path 를 입력해주세요
                train_data_loader(train_dataset_path,
                                  input_shape[:2],
                                  output_path)

        with open(output_path[0], 'rb') as img_f:
            img_list = pickle.load(img_f)
        with open(output_path[1], 'rb') as label_f:
            label_list = pickle.load(label_f)

        a_train = np.asarray(img_list)
        labels = np.asarray(label_list)
        a_train = a_train.astype('float32')
        a_train /= 255
        train_size = a_train.shape[0]
        train_gen = ThreadsafeIter(get_sample_generator({'a': a_train, 'y': labels},
                                                        batch_size=batch_size,
                                                        img_shape=input_shape,
                                                        model=get_feature_layer,
                                                        hard=True))
        total_train_samples = 0
        for label in set(labels):
            y_set = labels[labels == label]
            total_train_samples += len(list(itertools.permutations(y_set, 2)))
        print(train_size, 'train samples > ', total_train_samples)
        steps_per_epoch = int(np.ceil(total_train_samples / float(batch_size)))

        """ Pre-training data """
        x_train, x_test, y_train, y_test = train_test_split(a_train, labels,
                                                            test_size=opt.pretrain_test_split,
                                                            random_state=0)

        """ Callback """
        lr_scheduler = LearningRateScheduler(lr_schedule)
        reduce_lr = ReduceLROnPlateau(factor=np.sqrt(0.1), patience=5)
        callbacks = [reduce_lr, lr_scheduler]

        """ Pre-training base model first """
        if base_model is not None:
            train_datagen = ImageDataGenerator(rotation_range=40,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True,
                                               fill_mode='nearest')
            train_generator = train_datagen.flow(x_train, y_train, batch_size=opt.pretrain_batch_size)
            train_generator = ThreadsafeIter(train_generator)
            pretrain_steps_per_epoch = int(np.ceil(x_train.shape[0] / float(opt.pretrain_batch_size)))
            if x_test.size != 0:
                test_datagen = ImageDataGenerator()
                test_generator = test_datagen.flow(x_test, y_test, batch_size=opt.pretrain_batch_size)
                test_generator = ThreadsafeIter(test_generator)
                test_validation_steps = int(np.ceil(x_test.shape[0] / float(opt.pretrain_batch_size)))

            optm = keras.optimizers.Adam(lr_schedule(0))
            net = keras.layers.Dense(num_classes, activation='softmax')(base_model.output)
            pretrain = keras.models.Model(inputs=[base_model.input], outputs=net)
            if opt.num_gpus > 1:
                pretrain = ModelMGPU(pretrain, gpus=opt.num_gpus)
            pretrain.compile(loss='categorical_crossentropy',
                             optimizer=optm,
                             metrics=['accuracy'])
            pretrain.summary()
            res = pretrain.fit_generator(train_generator,
                                         epochs=opt.pretrain_n_epoch,
                                         steps_per_epoch=pretrain_steps_per_epoch,
                                         validation_data=test_generator if x_test.size != 0 else None,
                                         validation_steps=test_validation_steps if x_test.size != 0 else None,
                                         shuffle=True,
                                         workers=4,
                                         callbacks=callbacks)

        callbacks = []

        """ Training loop """
        for epoch in range(nb_epoch):
            res = model.fit_generator(generator=train_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      shuffle=True,
                                      callbacks=callbacks)
            print(res.history)
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
            nsml.save(epoch)
