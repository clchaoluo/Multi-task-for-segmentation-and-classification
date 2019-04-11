# img2pkl.py
from __future__ import division, print_function, absolute_import
import os
import sys
# import tarfile
import numpy as np
import pickle
import random
from PIL import Image


# 主要函数接口
def load_data(dirname="17flowers", pklpath='17flowers/jpg/',
              resize_pics=(224, 224), shuffle=True, one_hot=False):
    dataset_file = os.path.join(dirname, 'dataset.pkl')
    X, Y = build_image_dataset_from_dir(directory=pklpath,
                                        dataset_file=dataset_file,
                                        resize=resize_pics,
                                        filetypes=['.jpg', '.jpeg'],
                                        convert_gray=False,
                                        shuffle_data=shuffle,
                                        categorical_Y=one_hot)
    
    return X, Y


####
def build_image_dataset_from_dir(directory,
                                 dataset_file="dataset.pkl",
                                 resize=None, convert_gray=None,
                                 filetypes=None, shuffle_data=False,
                                 categorical_Y=False):
    try:
        X, Y = pickle.load(open(dataset_file, 'rb'))
    except Exception:
        X, Y = image_dirs_to_samples(directory, resize, convert_gray, filetypes)
        if categorical_Y:
            Y = to_categorical(Y, np.max(Y) + 1)  # First class is '0'
        if shuffle_data:
            X, Y = shuffle(X, Y)
        pickle.dump((X, Y), open(dataset_file, 'wb'))
    return X, Y


########
def image_dirs_to_samples(directory, resize=None, convert_gray=None,
                          filetypes=None):
    print("Starting to parse images...")
    if filetypes:
        if filetypes not in [list, tuple]: filetypes = list(filetypes)
    samples, targets = directory_to_samples(directory, flags=filetypes)
    for i, s in enumerate(samples):
        samples[i] = load_image(s)
        if resize:
            samples[i] = resize_image(samples[i], resize[0], resize[1])
        if convert_gray:
            samples[i] = convert_color(samples[i], 'L')
        samples[i] = pil_to_nparray(samples[i])
        samples[i] /= 255.
    print("Parsing Done!")
    return samples, targets


# 类别one-hot编码
def to_categorical(y, nb_classes):
    """ to_categorical.
    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.
    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


# 将数据打乱
def shuffle(*arrs):
    """ shuffle.
    Shuffle given arrays at unison, along first axis.
    Arguments:
        *arrs: Each array to shuffle at unison as a parameter.
    Returns:
        Tuple of shuffled arrays.
    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


# 遍历各文件夹和文件夹中的图片
def directory_to_samples(directory, flags=None):
    """ Read a directory, and list all subdirectories files as class sample """
    samples = []
    targets = []
    label = 0
    classes = sorted(os.walk(directory).next()[1])
    for c in classes:
        c_dir = os.path.join(directory, c)
        for sample in os.walk(c_dir).next()[2]:
            if not flags or any(flag in sample for flag in flags):
                samples.append(os.path.join(c_dir, sample))
                targets.append(label)
        label += 1
    return samples, targets


# 使用PIL库读取图片
def load_image(in_image):
    """ Load an image, returns PIL.Image. """
    img = Image.open(in_image)
    return img


# 图片尺度变换函数
def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    """ Resize an image.
    Arguments:
        in_image: `PIL.Image`. The image to resize.
        new_width: `int`. The image new width.
        new_height: `int`. The image new height.
        out_image: `str`. If specified, save the image to the given path.
        resize_mode: `PIL.Image.mode`. The resizing mode.
    Returns:
        `PIL.Image`. The resize image.
    """
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


#
def convert_color(in_image, mode):
    """ Convert image color with provided `mode`. """
    return in_image.convert(mode)


# 加载图片并转换为numpy多维数组
def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array. """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


if __name__ == "__main__":
    load_data(dirname="dataSet", resize_pics=(224, 224))