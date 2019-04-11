import math
import pathlib
import numpy as np
import SimpleITK as sitk


def calc_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = np.max(img1)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_snr():
    normal_path = pathlib.Path(r'/home/luochao/project/Data/Heart/MutiTask/predict')
    patient_path = pathlib.Path(r'/home/luochao/project/Data/Heart/MutiTask/Seg')
    summary_snr = pathlib.Path(r'/home/luochao/project/residual-attention-network_new_dcmdata/dataset/summary_psnr_80.csv')

    with summary_snr.open('a', encoding='utf-8') as f:
        snr_list = []
        
        for both_path in [normal_path, patient_path]:
            for image_path in both_path.iterdir():
                if image_path.stem.endswith('_seg'):
                    continue

                image = sitk.ReadImage(str(image_path))
                image = sitk.GetArrayFromImage(image)
                # _min = 0.
                # _max = 1.
                # image = (_max - _min) * (image - image.min()) / (image.max() - image.min()) + _min
                
                print(image_path.stem)
                file_name = image_path.stem
                
                label_path = image_path.with_name(file_name[:-13] + '_seg.mha')
                label = sitk.ReadImage(str(label_path))
                label = sitk.GetArrayFromImage(label)

                # only use myocardial label for classification
                label_type = 1
                print(label.shape)
                print(image.shape)
                myocardial = np.zeros_like(image)
                myocardial[np.where(label == label_type)] = image[np.where(label == label_type)]
                background = np.zeros_like(image)
                background[np.where(label != label_type)] = image[np.where(label != label_type)]

                snr = np.sum(myocardial) / (np.sum(background))
                psnr = calc_psnr(image, myocardial)
                print('{},{}'.format(image_path.name, psnr))
                f.write('{},{}\n'.format(image_path.name, psnr))

                snr_list.append(psnr)
        f.write('max,{}\n'.format(np.max(snr_list)))
        f.write('min,{}\n'.format(np.min(snr_list)))
        f.write('mean,{}\n'.format(np.mean(snr_list)))
        f.write('stddev,{}\n'.format(np.std(snr_list)))


if __name__ == '__main__':
    calc_snr()
