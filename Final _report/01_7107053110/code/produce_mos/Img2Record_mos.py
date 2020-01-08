from supplement import *
import tensorflow as tf
import os
import numpy as np
import skimage.io as io
import skimage.color as color


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


'''Define some parameters'''
# clustering number list
clu_number_list = [64]
# save name
name_list = ["mos_gray", "mos_color", "slic"]

# idr
input_dir = "./input/"
output_dir = "./output_1000/"
record_dir = "./record/"

# the path of training dataset record
path_tfrecords_train = os.path.join(record_dir, "train1000_mos2.tfrecords")
# To produce training record file
train_writer = tf.io.TFRecordWriter(path_tfrecords_train)


'''Collect the path of the images '''
input_paths = get_path(input_dir=input_dir)
print("total:", len(input_paths))

'''check output dir'''
check_output(output_dir)
for i in clu_number_list:
    check_output(output_dir + str(i))
    for j in name_list:
        check_output(output_dir + str(i) + "/{}/".format(j))
check_output(output_dir + "gray/")

check_output(record_dir)

count = 0

'''use the SLIC to construct a dictionary dataset'''
for img_path in input_paths:
    # read the image from path
    img = io.imread(img_path)
    # [0, 1] (float64) --> [0, 1] (float32)
    gray = np.array(color.rgb2gray(img), np.float32)

    # change the color space of image (RGB --> LAB)
    lab = np.array(color.rgb2lab(img), np.float32)

    # obtain the luminance of the image
    # [0:100] (float32) --> [-1,1] (float32)
    L_image = np.array(lab[:, :, 0] / 50 - 1, np.float32)

    # obtain the AB-color space of the image
    # [-128:127] (float32) --> [-1, 1] (float32)
    AB_image = np.array(lab[:, :, 1:] / 128, np.float32)

    # L_image Range: [-1, 1] (float32) --> [0, 1] (float32)
    L_image_adjust = (L_image + 1) / 2

    # AB_image Range: [-1, 1] (float32) --> [0, 1] (float32)
    AB_image_adjust = (AB_image + 1) / 2

    mosaImg_list = []
    for k in clu_number_list:
        model = SLIC(L_image_adjust, AB_image_adjust, k)

        # [0, 1] (float32) (because it produced from L_image_adjust) --> [-1, 1] (float32)
        mos_gray = (model.mosaImgGray*2) - 1
        mosaImg_list.append(mos_gray)
        # [0, 1] (float32) (because it produced from AB_image_adjust) --> [-1, 1] (float32)
        mos_ab = (model.mosaImgAB * 2) - 1
        mosaImg_list.append(mos_ab)

        # save gray scale image after clustering [0, 1] (float32) --> [0, 255] (uint8)
        io.imsave(output_dir + str(k) + "/slic/{}.png".format(count), trans_img(model.img))
        # save gray scale mos after clustering [0, 1] (float32) --> [0, 255] (uint8)
        io.imsave(output_dir + str(k) + "/mos_gray/{}.png".format(count), trans_img(model.mosaImgGray))
        # save gray+ab mos after clustering [0, 1] (float32) --> [0, 255] (uint8)
        io.imsave(output_dir + str(k) + "/mos_color/{}.png".format(count), cat_trans_img(model.mosaImgGray, mos_ab))

    # save gray scale image [0, 1] (float32) --> [0, 255] (uint8)
    io.imsave(output_dir + "gray/{}.png".format(count), trans_img(gray))

    count += 1
    print(count)

    '''original images --> Feature --> Features --> Example --> TFRecord'''
    '''get raw'''
    L_image_raw = L_image.tostring()
    AB_image_raw = AB_image.tostring()
    Mos_gray64_raw = mosaImg_list[0].tostring()
    Mos_ab64_raw = mosaImg_list[1].tostring()

    # feature
    example = tf.train.Example(features=tf.train.Features(feature={
        'L_image_raw': _bytes_feature(L_image_raw),
        'AB_image_raw': _bytes_feature(AB_image_raw),
        'Mos_gray64_raw': _bytes_feature(Mos_gray64_raw),
        'Mos_ab64_raw': _bytes_feature(Mos_ab64_raw)}))
    train_writer.write(example.SerializeToString())

train_writer.close()


