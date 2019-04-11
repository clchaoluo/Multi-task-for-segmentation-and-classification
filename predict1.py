# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import utils
import xml.dom.minidom as xmldom
import SimpleITK as sitk
import numpy as np




os.environ['CUDA_VISIBLE_DEVICES']='6'


def compute_metrics(name,num):
    seg_tail = r"_seg.mha"
    predict_path = r"/home/luochao/project/Data/New_DCM_Patient_128_1/predict"+str(num)+"/" + name + 'PredictResult.mha'
    truth_path = r"/home/luochao/project/Data/New_DCM_Patient_128_1/allSeg2/" + name + seg_tail
    output_path = r"/home/luochao/project/Data/New_DCM_Patient_128_1/score"+str(num)+"/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # os.chdir(score_dir)
    output = os.path.join(output_path, str(name) + ".xml")
    os.system("./EvaluateSegmentation " + truth_path + " " + predict_path + " –use all " + " -xml " + output)
    # parse_xml()
    print('******done*****')
def parse_xml(num):
    output_path = '/home/luochao/project/Data/New_DCM_Patient_128_1/score'+str(num)+'/'
    if not os.path.exists('/home/luochao/project/Data/New_DCM_Patient_128_1/DiceResult'):
        os.mkdir('/home/luochao/project/Data/New_DCM_Patient_128_1/DiceResult')
    result_path = '/home/luochao/project/Data/New_DCM_Patient_128_1/DiceResult/result'+str(num)+'.txt'
    
    file_txt = open(result_path,'w')
    xml_list = os.listdir(output_path)
    print('len:', len(xml_list))
    metrics = ['DICE','JACRD','AUC','SNSVTY','PRCISON','FMEASR']
    count = 0
    for metric in metrics:
        avg_value = 0
        for xml in xml_list:
            # 得到文档对象
            print('xml', xml)
            domobj = xmldom.parse(os.path.join(output_path,xml))
            # 得到元素对象
            elementobj = domobj.documentElement
            # 获得子标签
            subElementObj = elementobj.getElementsByTagName("%s"%metric)
            # 获得标签属性值
            print('count', count)
            name = subElementObj[0].getAttribute("name")
            value = subElementObj[0].getAttribute("value")
            avg_value += float(value)
            count += 1
        file_txt.write(str(name)+' '+str(avg_value/len(xml_list))+'\n')
    return
def comput_cut(main_path, data):
    CT_predict_array1 = data[1]
    for b in range(76):
        for c in range(76):
            if CT_predict_array1[b][c] > 0.2:
                CT_predict_array1[b][c] = 1
            else:
                CT_predict_array1[b][c] = 0
    CT_predict_array = CT_predict_array1
    CT_predict_array = CT_predict_array.reshape(1, 76, 76)
    CT_predict_itk = sitk.GetImageFromArray(CT_predict_array)
    test_list = open('/home/luochao/project/residual-attention-network/test_cut.txt')
    test_list = [line.strip() for line in test_list.readlines()]
    patient_name = test_list[k]
    patient_name = os.path.splitext(patient_name)[0]
    # patient_name = 'cai_shang_wei_T1_Series0013'
    CT_itk = sitk.ReadImage('/home/luochao/project/Data/Heart/MutiTask/cut/allOri/' + patient_name + '.mha')
    CT_predict_itk.SetOrigin(CT_itk.GetOrigin())
    CT_predict_itk.SetSpacing(CT_itk.GetSpacing())
    CT_predict_itk.SetDirection(CT_itk.GetDirection())
    sitk.WriteImage(CT_predict_itk, main_path + '/cutPredict/' + patient_name + 'PredictResult.mha')
    print('done computing dice')
    compute_metrics(patient_name)


if __name__ == '__main__':
    num = 1
    main_path = '/home/luochao/project/Data/New_DCM_Patient_128_1'
    train_X, train_y_class, train_y_seg, valid_X, valid_y_class, valid_y_seg, test_X, test_y, test_class_y = utils.load_data_new_dcm_128(num)
    # x = tf.placeholder(tf.float32, [None, 128, 128, 1])
    # y = tf.placeholder(tf.float32, [None, 128, 128, 1])
    HOME_DIR = os.environ['HOME'] + '/project/residual-attention-network_new_dcmdata/trained_models_'+str(num)+'/'
    # predict
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(HOME_DIR + 'model.ckpt-77.meta')
        saver.restore(sess, HOME_DIR + 'model.ckpt-77')  # .data文件
        graph = tf.get_default_graph()
        
        x = graph.get_operation_by_name('x').outputs[0]
        y = tf.get_collection("pred_network")[0]
        # r = 142
        r = 90
        for k in range(r):
            result = sess.run(y, feed_dict={x: test_X[k:k+1]})
            print()
            data = result[0]
            data = np.transpose(data, (2, 1, 0))
            cut = 0
            if cut == 1:
                comput_cut(main_path, data)
            else:
                # 分3类
                
                # CT_predict_array = data[1]
                # CT_predict_array0 = np.zeros((128,128))
                # CT_predict_array1 = np.ones((128,128))
    
                # 修改
                # CT_predict_array0 = data[0]
                # for b in range(128):
                #     for c in range(128):
                #         if CT_predict_array0[b][c] > 0.2:
                #             CT_predict_array0[b][c] = 1
                #         else:
                #             CT_predict_array0[b][c] = 0
                #
                #
                # 修改
                # CT_predict_array = np.zeros((76,76))
                # CT_predict_array1 = data[1]
                # for b in range(76):
                #     for c in range(76):
                #         if CT_predict_array1[b][c] > 0.5:
                #             CT_predict_array1[b][c] = 1
                #         else:
                #             CT_predict_array1[b][c] = 0
                #
                # # 修改
                # CT_predict_array2 = data[2]
                # for b in range(76):
                #     for c in range(76):
                #         if CT_predict_array2[b][c] > 0.8:
                #             CT_predict_array2[b][c] = 2
                #         else:
                #             CT_predict_array2[b][c] = 0
                #
                # CT_predict_array = CT_predict_array1 + CT_predict_array2
                # for b in range(76):
                #     for c in range(76):
                #         if CT_predict_array[b][c] == 2:
                #             CT_predict_array[b][c] = 0

                CT_predict_array = data[1]
                for b in range(128):
                    for c in range(128):
                        if CT_predict_array[b][c] > 0.3:
                            CT_predict_array[b][c] = 1
                        else:
                            CT_predict_array[b][c] = 0

                CT_predict_array = CT_predict_array.reshape(1, 128, 128)
                CT_predict_itk = sitk.GetImageFromArray(CT_predict_array)
                test_list = open(main_path + '/test_cut'+str(num)+'.txt')
                test_list = [line.strip() for line in test_list.readlines()]
                patient_name = test_list[k]
                patient_name = os.path.splitext(patient_name)[0]
                # patient_name = 'cai_shang_wei_T1_Series0013'
                CT_itk = sitk.ReadImage(main_path+'/allOri/'+patient_name+'.mha')
                CT_predict_itk.SetOrigin(CT_itk.GetOrigin())
                CT_predict_itk.SetSpacing(CT_itk.GetSpacing())
                CT_predict_itk.SetDirection(CT_itk.GetDirection())
                predict_path = main_path+'/predict'+str(num)+'/'
                if not os.path.exists(predict_path):
                    os.mkdir(predict_path)
                sitk.WriteImage(CT_predict_itk, predict_path + patient_name + 'PredictResult.mha')
                print('done computing dice')
                List = ['lu_shao_hui_0001']
                if patient_name not in List:
                    compute_metrics(patient_name,num)
        parse_xml(num)