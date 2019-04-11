# -*- coding: utf-8 -*-
"""
utils file
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from hyperparameter import HyperParams as hp
import math
import errno
import shutil


def load_data():
    if hp.target_dataset == "CIFAR-10":
        if os.path.exists(hp.DATASET_DIR + hp.target_dataset):
            print("load data from pickle")
            with open(hp.DATASET_DIR + hp.target_dataset + "/train_X.pkl", 'rb') as f:
                train_X = pickle.load(f)
            with open(hp.DATASET_DIR + hp.target_dataset + "/train_y.pkl", 'rb') as f:
                train_y = pickle.load(f)
            with open(hp.DATASET_DIR + hp.target_dataset + "/valid_X.pkl", 'rb') as f:
                valid_X = pickle.load(f)
            with open(hp.DATASET_DIR + hp.target_dataset + "/valid_y.pkl", 'rb') as f:
                valid_y = pickle.load(f)
            with open(hp.DATASET_DIR + hp.target_dataset + "/test_X.pkl", 'rb') as f:
                test_X = pickle.load(f)
            with open(hp.DATASET_DIR + hp.target_dataset + "/test_y.pkl", 'rb') as f:
                test_y = pickle.load(f)
        else:
            (cifar_X_1, cifar_y_1), (cifar_X_2, cifar_y_2) = cifar10.load_data()
            cifar_X = np.r_[cifar_X_1, cifar_X_2]
            cifar_y = np.r_[cifar_y_1, cifar_y_2]

            cifar_X = cifar_X.astype('float32') / 255.0
            cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

            train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y, test_size=5000,
                                                                random_state=hp.RANDOM_STATE)
            train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=5000,
                                                                  random_state=hp.RANDOM_STATE)

            os.mkdir(hp.DATASET_DIR + hp.target_dataset)
            with open(hp.DATASET_DIR + hp.target_dataset + "/train_X.pkl", 'wb') as f1:
                pickle.dump(train_X, f1)
            with open(hp.DATASET_DIR + hp.target_dataset + "/train_y.pkl", 'wb') as f1:
                pickle.dump(train_y, f1)
            with open(hp.DATASET_DIR + hp.target_dataset + "/valid_X.pkl", 'wb') as f1:
                pickle.dump(valid_X, f1)
            with open(hp.DATASET_DIR + hp.target_dataset + "/valid_y.pkl", 'wb') as f1:
                pickle.dump(valid_y, f1)
            with open(hp.DATASET_DIR + hp.target_dataset + "/test_X.pkl", 'wb') as f1:
                pickle.dump(test_X, f1)
            with open(hp.DATASET_DIR + hp.target_dataset + "/test_y.pkl", 'wb') as f1:
                pickle.dump(test_y, f1)

    return train_X, train_y, valid_X, valid_y, test_X, test_y

def load_data_heart():
    data_path = '/home/luochao/project/Data/Heart/PKL'
    if os.path.exists(data_path):
        print("load data from pickle heart")
        with open(data_path + "/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open(data_path + "/train_y.pkl", 'rb') as f:
            train_y = pickle.load(f)
        with open(data_path + "/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open(data_path + "/valid_y.pkl", 'rb') as f:
            valid_y = pickle.load(f)
        with open(data_path + "/test_X.pkl", 'rb') as f:
            test_X = pickle.load(f)
        with open(data_path + "/test_y.pkl", 'rb') as f:
            test_y = pickle.load(f)
    return train_X, train_y, valid_X, valid_y, test_X, test_y
def load_data_ori_heart():
    data_path = '/home/luochao/project/Data/Heart/AllOri/Pkl'
    if os.path.exists(data_path):
        print("load data from pickle ori heart")
        with open(data_path + "/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open(data_path + "/train_y.pkl", 'rb') as f:
            train_y = pickle.load(f)
        with open(data_path + "/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open(data_path + "/valid_y.pkl", 'rb') as f:
            valid_y = pickle.load(f)
        with open(data_path + "/test_X.pkl", 'rb') as f:
            test_X = pickle.load(f)
        with open(data_path + "/test_y.pkl", 'rb') as f:
            test_y = pickle.load(f)
    return train_X, train_y, valid_X, valid_y, test_X, test_y

# 半折法取数据
def load_data_half_heart(j):
    data_path = '/home/luochao/project/Data/Heart/PKL/half_validation/'+str(j)+'_Pkl'
    if os.path.exists(data_path):
        print("load data from pickle ori heart")
        with open(data_path + "/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open(data_path + "/train_y.pkl", 'rb') as f:
            train_y = pickle.load(f)
        with open(data_path + "/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open(data_path + "/valid_y.pkl", 'rb') as f:
            valid_y = pickle.load(f)
        # with open(data_path + "/test_X.pkl", 'rb') as f:
        #     test_X = pickle.load(f)
        # with open(data_path + "/test_y.pkl", 'rb') as f:
        #     test_y = pickle.load(f)
    # return train_X, train_y, valid_X, valid_y, test_X, test_y
    return train_X, train_y, valid_X, valid_y

def load_data_ori_cut_heart():
    data_path = '/home/luochao/project/Data/Heart/AllOri/CutPkl'
    if os.path.exists(data_path):
        print("load data from pickle cut ori heart")
        with open(data_path + "/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open(data_path + "/train_y.pkl", 'rb') as f:
            train_y = pickle.load(f)
        with open(data_path + "/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open(data_path + "/valid_y.pkl", 'rb') as f:
            valid_y = pickle.load(f)
        with open(data_path + "/test_X.pkl", 'rb') as f:
            test_X = pickle.load(f)
        with open(data_path + "/test_y.pkl", 'rb') as f:
            test_y = pickle.load(f)
    return train_X, train_y, valid_X, valid_y, test_X, test_y
def load_data_Muti_heart():
    data_path = '/home/luochao/project/Data/Heart/MutiTask'
    if os.path.exists(data_path):
        print("load data from pickle Muti heart")
        with open(data_path + "/ClassPkl/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open(data_path + "/ClassPkl/train_y.pkl", 'rb') as f:
            train_y_class = pickle.load(f)
        with open(data_path + "/SegPkl/train_y.pkl", 'rb') as f:
            train_y_seg = pickle.load(f)
        
        with open(data_path + "/ClassPkl/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open(data_path + "/ClassPkl/valid_y.pkl", 'rb') as f:
            valid_y_class = pickle.load(f)
        with open(data_path + "/SegPkl/valid_y.pkl", 'rb') as f:
            valid_y_seg = pickle.load(f)
        with open(data_path + "/SegPkl/test_X.pkl", 'rb') as f:
            test_X = pickle.load(f)
        with open(data_path + "/SegPkl/test_y.pkl", 'rb') as f:
            test_y = pickle.load(f)
        
    return train_X, train_y_class, train_y_seg, valid_X, valid_y_class, valid_y_seg, test_X, test_y


def load_data_Muti_heart_cut():
    data_path = '/home/luochao/project/Data/Heart/MutiTask/cut'
    if os.path.exists(data_path):
        print("load data from pickle Muti heart cut")
        with open(data_path + "/ClassPkl/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open(data_path + "/ClassPkl/train_y.pkl", 'rb') as f:
            train_y_class = pickle.load(f)
        with open(data_path + "/SegPkl/train_y.pkl", 'rb') as f:
            train_y_seg = pickle.load(f)
        
        with open(data_path + "/ClassPkl/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open(data_path + "/ClassPkl/valid_y.pkl", 'rb') as f:
            valid_y_class = pickle.load(f)
        with open(data_path + "/SegPkl/valid_y.pkl", 'rb') as f:
            valid_y_seg = pickle.load(f)
        with open(data_path + "/SegPkl/test_X.pkl", 'rb') as f:
            test_X = pickle.load(f)
        with open(data_path + "/SegPkl/test_y.pkl", 'rb') as f:
            test_y = pickle.load(f)
    
    return train_X, train_y_class, train_y_seg, valid_X, valid_y_class, valid_y_seg, test_X, test_y


def load_data_Muti_heart_new_class():
    data_path = '/home/luochao/project/Data/classification'
    if os.path.exists(data_path):
        print("load data from pickle Muti heart cut")
        with open(data_path + "/ClassPkl/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open(data_path + "/ClassPkl/train_y.pkl", 'rb') as f:
            train_y_class = pickle.load(f)
        with open(data_path + "/ClassPkl/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open(data_path + "/ClassPkl/valid_y.pkl", 'rb') as f:
            valid_y_class = pickle.load(f)
    return train_X, train_y_class, valid_X, valid_y_class


def load_data_new_and_old_128(num):
    data_path = '/home/luochao/project/Data/NewAndOldHearData/PKL'+str(num)
    if os.path.exists(data_path):
        print("load data from pickle Muti heart cut")
        with open(data_path + "/ClassPkl/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open(data_path + "/ClassPkl/train_y.pkl", 'rb') as f:
            train_y_class = pickle.load(f)
        with open(data_path + "/SegPkl/train_y.pkl", 'rb') as f:
            train_y_seg = pickle.load(f)
        
        with open(data_path + "/ClassPkl/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open(data_path + "/ClassPkl/valid_y.pkl", 'rb') as f:
            valid_y_class = pickle.load(f)
        with open(data_path + "/SegPkl/valid_y.pkl", 'rb') as f:
            valid_y_seg = pickle.load(f)
            
        with open(data_path + "/SegPkl/test_X.pkl", 'rb') as f:
            test_X = pickle.load(f)
        with open(data_path + "/SegPkl/test_y.pkl", 'rb') as f:
            test_y = pickle.load(f)
    
    return train_X, train_y_class, train_y_seg, valid_X, valid_y_class, valid_y_seg, test_X, test_y


def load_data_new_dcm_128(num):
    data_path = '/home/luochao/project/Data/classification/PKL' + str(num)
    if os.path.exists(data_path):
        print("load data from pickle Muti heart cut")
        with open(data_path + "/ClassPkl/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open(data_path + "/ClassPkl/train_y.pkl", 'rb') as f:
            train_y_class = pickle.load(f)
        with open(data_path + "/SegPkl/train_y.pkl", 'rb') as f:
            train_y_seg = pickle.load(f)
        
        with open(data_path + "/ClassPkl/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open(data_path + "/ClassPkl/valid_y.pkl", 'rb') as f:
            valid_y_class = pickle.load(f)
        with open(data_path + "/SegPkl/valid_y.pkl", 'rb') as f:
            valid_y_seg = pickle.load(f)
        
        with open(data_path + "/SegPkl/test_X.pkl", 'rb') as f:
            test_X = pickle.load(f)
        with open(data_path + "/SegPkl/test_y.pkl", 'rb') as f:
            test_y = pickle.load(f)
        with open(data_path + "/ClassPkl/test_y.pkl", 'rb') as f:
            test_class_y = pickle.load(f)
    
    return train_X, train_y_class, train_y_seg, valid_X, valid_y_class, valid_y_seg, test_X, test_y, test_class_y


def load_data_new_dcm_128_updown(num):
    data_path = '/home/luochao/project/Data/New_DCM_Patient_128_1_updown/PKL' + str(num)
    if os.path.exists(data_path):
        print("load data from pickle Muti heart cut")
        with open(data_path + "/ClassPkl/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open(data_path + "/ClassPkl/train_y.pkl", 'rb') as f:
            train_y_class = pickle.load(f)
        with open(data_path + "/SegPkl/train_y.pkl", 'rb') as f:
            train_y_seg = pickle.load(f)
        
        with open(data_path + "/ClassPkl/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open(data_path + "/ClassPkl/valid_y.pkl", 'rb') as f:
            valid_y_class = pickle.load(f)
        with open(data_path + "/SegPkl/valid_y.pkl", 'rb') as f:
            valid_y_seg = pickle.load(f)
        
        with open(data_path + "/SegPkl/test_X.pkl", 'rb') as f:
            test_X = pickle.load(f)
        with open(data_path + "/SegPkl/test_y.pkl", 'rb') as f:
            test_y = pickle.load(f)
        with open(data_path + "/ClassPkl/test_y.pkl", 'rb') as f:
            test_class_y = pickle.load(f)
    
    return train_X, train_y_class, train_y_seg, valid_X, valid_y_class, valid_y_seg, test_X, test_y, test_class_y

def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]
    
    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print ('Warning: {}'.format(e))


def create_dir(path):
    """
    Creates a directory
    :param path: string
    :return: nothing
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """
    if not os.path.exists(path):
        create_dir(path)
    
    if empty:
        empty_dir(path)
