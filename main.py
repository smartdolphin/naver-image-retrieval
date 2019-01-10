# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import h5py
import argparse
import pickle

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from data_loader import triplet_train_data_loader
from network import get_model


def bind_model(model):
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

        get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-2].output])

        print('inference start')

        # inference
        query_vecs = get_feature_layer([query_img, 0])[0]

        # caching db output, db inference
        db_output = './db_infer.pkl'
        if os.path.exists(db_output):
            with open(db_output, 'rb') as f:
                reference_vecs = pickle.load(f)
        else:
            reference_vecs = get_feature_layer([reference_img, 0])[0]
            with open(db_output, 'wb') as f:
                pickle.dump(reference_vecs, f)

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


def get_sample_generator(ds, batch_size, hard=False, raise_stop_event=False):
    left, limit = 0, ds['a'].shape[0]
    while True:
        right = min(left + batch_size, limit)

        batch_inds = list(np.arange(left, right))
        nagative_inds = []
        for i in batch_inds:
            _y = np.argmax(ds['y'], axis=1)[i]
            mask = np.argmax(ds['y'], axis=1) == _y
            nagative_inds.append(np.random.choice(np.where(np.logical_not(mask))[0], size=1)[0])

        X = [ds[t][left:right, :] for t in ['a', 'p']] + [ds['a'][nagative_inds]]
        Y = ds['y'][left:right]
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
    args.add_argument('--batch_size', type=int, default=128)

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
    model = get_model('triplet', 224, num_classes)
    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate RMSprop optimizer """
        opt = keras.optimizers.rmsprop(lr=0.00045, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        """ Load data """
        print('dataset path', DATASET_PATH)
        output_path = './data.h5py'
        train_dataset_path = DATASET_PATH + '/train/train_data'

        if nsml.IS_ON_NSML:
            # Caching file
            if os.path.exists(output_path) is False:
                nsml.cache(triplet_train_data_loader,
                           data_path=train_dataset_path,
                           img_size=input_shape[:2],
                           output_path=output_path,
                           num_classes=num_classes,
                           train_ratio=1.0)
        else:
            # local에서 실험할경우 dataset의 local-path 를 입력해주세요
            if os.path.exists(output_path) is False:
                triplet_train_data_loader(train_dataset_path,
                                          input_shape[:2],
                                          output_path=output_path,
                                          num_classes=num_classes,
                                          train_ratio=1.0)
        data = h5py.File(output_path, 'r')
        train = data['train']
        dev = data['dev']
        total_train_samples = train['y'].shape[0]
        train_gen = get_sample_generator(train, batch_size=batch_size)
        steps_per_epoch = int(np.ceil(total_train_samples / float(batch_size)))

        total_dev_samples = dev['y'].shape[0]
        dev_gen = get_sample_generator(dev, batch_size=batch_size)
        validation_steps = int(np.ceil(total_dev_samples / float(batch_size)))

        train_size = train['a'].shape[0]
        a_train = train['a'].value.reshape(train_size, 224, 224, 3)
        a_train /= 255
        p_train = train['p'].value.reshape(train_size, 224, 224, 3)
        p_train /= 255
        print(train_size, 'train samples')

        """ Callback """
        monitor = 'acc'
        callbacks = [ReduceLROnPlateau(monitor=monitor, patience=3)]


        """ Training loop """
        for epoch in range(nb_epoch):
            res = model.fit_generator(generator=train_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epoch+1,
                                      validation_data=dev_gen,
                                      validation_steps=validation_steps,
                                      shuffle=True,
                                      callbacks=callbacks)
            print(res.history)
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
            nsml.save(epoch)
        data.close()
