# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import h5py
import tqdm
import pickle
import itertools
import numpy as np


def train_data_loader(data_path, img_size, output_path):
    label_list = []
    img_list = []
    label_idx = 0

    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        for filename in files:
            img_path = os.path.join(root, filename)
            try:
                img = cv2.imread(img_path, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
            except:
                continue
            label_list.append(label_idx)
            img_list.append(img)
        label_idx += 1

    # write output file for caching
    with open(output_path[0], 'wb') as img_f:
        pickle.dump(img_list, img_f)
    with open(output_path[1], 'wb') as label_f:
        pickle.dump(label_list, label_f)


def triplet_train_data_loader(data_path, img_size, output_path, num_classes, train_ratio=1.0):
    label_list = []
    a_img_list = []
    p_img_list = []
    y_vocab = {}
    label_idx = 0

    file_cnt = sum([len(list(itertools.permutations(files, 2)))
                    for _, _, files in os.walk(data_path) if files is not None])
    with tqdm.tqdm(total=file_cnt) as pbar:
        for root, dirs, files in os.walk(data_path):
            if not files:
                continue
            for a, p in itertools.permutations(files, 2):
                a_path = os.path.join(root, a)
                p_path = os.path.join(root, p)
                try:
                    a_img = cv2.imread(a_path, 1)
                    a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
                    a_img = cv2.resize(a_img, img_size)

                    p_img = cv2.imread(p_path, 1)
                    p_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
                    p_img = cv2.resize(p_img, img_size)
                except:
                    continue
                y_vocab[label_idx] = int(root.split('/')[-1])
                label_list.append(label_idx)
                a_img_list.append(a_img.reshape(-1))
                p_img_list.append(p_img.reshape(-1))
                pbar.update(1)
            label_idx += 1

    # save train/dev dataset
    with open('meta', 'wb') as meta_f:
        pickle.dump(y_vocab, meta_f)
    del y_vocab

    total_size = len(label_list)
    train_indices, train_size, dev_indices, dev_size = get_indices(total_size, train_ratio)

    data_fout = h5py.File(output_path, 'w')
    train = data_fout.create_group('train')
    dev = data_fout.create_group('dev')
    create_dataset(train, train_size, img_size[0], num_classes)
    create_dataset(dev, dev_size, img_size[0], num_classes)
    copy_dataset(train, {'a': np.asarray(a_img_list)[train_indices],
                         'p': np.asarray(p_img_list)[train_indices],
                         'y': np.asarray(label_list)[train_indices]},
                         num_classes)
    copy_dataset(dev, {'a': np.asarray(a_img_list)[dev_indices],
                       'p': np.asarray(p_img_list)[dev_indices],
                       'y': np.asarray(label_list)[dev_indices]},
                       num_classes)
    data_fout.close()
    print('train_size ~ %s, dev_size ~ %s' % (train_size, dev_size))


def triplet_data_loader(data_path, img_size, output_path, num_classes, train_ratio=1.0):
    label_list = []
    a_img_list = []
    y_vocab = {}
    label_idx = 0

    file_cnt = sum([len(files) for _, _, files in os.walk(data_path) if files is not None])
    with tqdm.tqdm(total=file_cnt) as pbar:
        for root, dirs, files in os.walk(data_path):
            if not files:
                continue
            for a in files:
                a_path = os.path.join(root, a)
                try:
                    a_img = cv2.imread(a_path, 1)
                    a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
                    a_img = cv2.resize(a_img, img_size)
                except:
                    continue
                y_vocab[label_idx] = int(root.split('/')[-1])
                label_list.append(label_idx)
                a_img_list.append(a_img.reshape(-1))
                pbar.update(1)
            label_idx += 1

    # save train/dev dataset
    with open('meta', 'wb') as meta_f:
        pickle.dump(y_vocab, meta_f)
    del y_vocab

    total_size = len(label_list)
    train_indices, train_size, dev_indices, dev_size = get_indices(total_size, train_ratio)

    data_fout = h5py.File(output_path, 'w')
    train = data_fout.create_group('train')
    dev = data_fout.create_group('dev')
    create_dataset(train, train_size, img_size[0], num_classes)
    create_dataset(dev, dev_size, img_size[0], num_classes)
    copy_dataset(train, {'a': np.asarray(a_img_list)[train_indices],
                         'y': np.asarray(label_list)[train_indices]},
                         num_classes)
    copy_dataset(dev, {'a': np.asarray(a_img_list)[dev_indices],
                       'y': np.asarray(label_list)[dev_indices]},
                       num_classes)
    data_fout.close()
    print('train_size ~ %s, dev_size ~ %s' % (train_size, dev_size))


def get_indices(size, train_ratio):
    tmp = np.random.rand(size)
    train_indices = tmp < train_ratio
    dev_indices = tmp >= train_ratio
    train_size = int(np.count_nonzero(train_indices))
    dev_size = int(np.count_nonzero(dev_indices))
    return train_indices, train_size, dev_indices, dev_size


def create_dataset(g, size, img_size, num_classes):
    shape = (size, img_size * img_size * 3)
    g.create_dataset('a', shape, dtype=np.float32)
    #g.create_dataset('p', shape, dtype=np.float32)
    g.create_dataset('y', (size,), dtype=np.int32)


def copy_dataset(dst, src, num_classes):
    dst['a'][:,:] = src['a'][:]
    #dst['p'][:,:] = src['p'][:]
    dst['y'][:] = src['y'][:]


# nsml test_data_loader
def test_data_loader(data_path):
    data_path = os.path.join(data_path, 'test', 'test_data')

    # return full path
    queries_path = [os.path.join(data_path, 'query', path) for path in os.listdir(os.path.join(data_path, 'query'))]
    references_path = [os.path.join(data_path, 'reference', path) for path in
                       os.listdir(os.path.join(data_path, 'reference'))]

    return queries_path, references_path


if __name__ == '__main__':
    query, refer = test_data_loader('./')
    print(query)
    print(refer)
