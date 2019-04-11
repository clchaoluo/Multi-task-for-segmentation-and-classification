import pathlib
import configparser

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from keras.models import Model

from models import xvgg16
from util import normalize


def normalize_array(arr):
    arr = 255.0 * (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def create_activation_map_for_image(model, image, pathology):
    image_preproc = np.expand_dims(image, 0)
    pathology_preproc = pathology.reshape(1, -1)
    pred = model.predict([image_preproc, pathology_preproc])
    print('Shape of predictions: {}'.format(pred.shape))

    img_orig = image
    heatmap = pred[0]

    # Uncomment it to emulate RELU activation
    # heatmap[heatmap < 0] = 0.

    ch0 = np.zeros_like(heatmap[:, :, 0])
    ch1 = np.zeros_like(heatmap[:, :, 0])
    ch2 = np.zeros_like(heatmap[:, :, 0])

    # Find how often maximum is in each pixel.
    for k in range(heatmap.shape[2]):
        p = heatmap[:, :, k]
        mx = p.max()
        if mx == 0:
            continue
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                if p[i, j] == mx:
                    ch0[i, j] += 1
                    ch2[i, j] += 1

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            mn = heatmap[i, j].min()
            mx = heatmap[i, j].max()
            mean = heatmap[i, j].mean()
            std = heatmap[i, j].std()
            # print(i, j, mn, mx, mean, std, mx - mn)
            ch1[i, j] = std

    ch0 = normalize_array(ch0)
    ch1 = normalize_array(ch1)
    ch2 = normalize_array(ch2)
    ch = np.stack((ch0, ch1, ch2), axis=2)

    ch = cv2.resize(ch.astype(np.uint8), (img_orig.shape[1], img_orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    ch = normalize_array(ch)

    heat = (0.2 * img_orig + (0.1 * img_orig * ch) / 255 + 0.7 * ch).astype(np.uint8)
    heat = heat.astype(np.uint8)
    heat = normalize_array(heat)

    gray_to3ch = np.stack((img_orig[:, :, 0],) * 3, axis=-1)

    return gray_to3ch, ch, heat


def create_activation_maps(cfp):
    n_classes = cfp.getint('Number', 'n_classes')
    re_img_width = cfp.getint('Number', 're_img_width')
    re_img_height = cfp.getint('Number', 're_img_height')
    n_channels = cfp.getint('Number', 'n_channels')
    input_image_shape = (re_img_width, re_img_height, n_channels)
    # number of all pathological features is 14
    pathology_fields = cfp.get('Suffix', 'pathology_fields').split(',')
    n_patholog_fields = len(pathology_fields)
    input_pathology_shape = (n_patholog_fields,)
    model = xvgg16(input_image_shape, input_pathology_shape, n_classes)
    model.load_weights('/home/xcy/results/dcm_class/event/cam/xvgg16_cam2/weights/best_weight.hdf5')

    test_result_path = pathlib.Path('/home/xcy/results/dcm_class/event/cam/xvgg16_cam2/class_predict.csv')
    test_result_df = pd.read_csv(test_result_path, header=None)
    # header: {name},{truth},{predict},{score_0},{score_1}
    for p_name in test_result_df[0]:
        if test_result_df.loc[test_result_df[0] == p_name, 1].values[0] == 0:
            ftype = 'alive'
        else:
            ftype = 'dead'
        # p_name = 'chen_sheng_lian_0004'
        ori_file_path = r'/home/xcy/dataset/dcm_data/class/event/classification_128x128/{}/{}.mha'.format(ftype, p_name)
        ori_file_path = pathlib.Path(ori_file_path)
        print(p_name)

        image = sitk.ReadImage(str(ori_file_path))
        image = sitk.GetArrayFromImage(image)
        image = np.transpose(image, (2, 1, 0))

        # used pathological data fields list
        pathology_fields = cfp.get('Suffix', 'pathology_fields').split(',')
        pathological_df = pd.read_csv(cfp.get('Path', 'pathological_file'))
        # do not use zh name
        pathological_df.drop(columns=['name'], inplace=True)
        # get pathological data
        pinyin_name = '_'.join([x for x in p_name.split('_') if not x.isnumeric()])
        pathology = pathological_df.loc[pathological_df['pinyin'] == pinyin_name]
        # data but exclude pinyin field
        pathology = pathology.loc[:, pathology_fields].values[0]

        model_modified = Model(inputs=model.inputs, outputs=model.get_layer(name='conv2d_1').output)

        gray_to3ch, ch, heat = create_activation_map_for_image(model_modified, image, pathology)

        plt.subplot(141)
        plt.axis('off')
        plt.imshow(gray_to3ch[:, :, 0], cmap='gray')

        plt.subplot(142)
        plt.axis('off')
        plt.imshow(cv2.cvtColor(ch.astype(np.uint8), cv2.COLOR_BGR2RGB))

        plt.subplot(143)
        plt.axis('off')
        plt.imshow(gray_to3ch[:, :, 0], cmap='gray')
        plt.imshow(cv2.cvtColor(ch.astype(np.uint8), cv2.COLOR_BGR2RGB), alpha=0.5)

        plt.subplot(144)
        plt.axis('off')
        plt.imshow(gray_to3ch[:, :, 0], cmap='gray')
        plt.imshow(cv2.cvtColor(ch.astype(np.uint8), cv2.COLOR_BGR2GRAY), cmap='jet', alpha=0.5)

        plt.savefig('tttt/cam_{}_{}.png'.format(ftype, p_name))
        # plt.show()


if __name__ == '__main__':
    _cfp = configparser.ConfigParser()
    _cfp.read('../config.ini')

    create_activation_maps(_cfp)
