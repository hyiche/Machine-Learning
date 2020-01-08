import tensorflow as tf
import os

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

record_dir = "./input/"
path_tfrecords_train = os.path.join(record_dir, "train400_mos2.tfrecords")
path_tfrecords_test = os.path.join(record_dir, "test.tfrecords")


'''TFRecord > Example > Features > Feature > original images'''
def read_and_decode(filename_queue):
    # Use tf.TFRecordReader() to read the TFRecord file and get the Example
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'L_image_raw': tf.FixedLenFeature([], tf.string),
        'AB_image_raw': tf.FixedLenFeature([], tf.string),
        'Mos_gray64_raw': tf.FixedLenFeature([], tf.string),
        'Mos_ab64_raw': tf.FixedLenFeature([], tf.string)})

    L_image = tf.decode_raw(features['L_image_raw'], tf.float32)
    AB_image = tf.decode_raw(features['AB_image_raw'], tf.float32)
    Mos_gray64 = tf.decode_raw(features['Mos_gray64_raw'], tf.float32)
    Mos_ab64 = tf.decode_raw(features['Mos_ab64_raw'], tf.float32)


    image_shape1 = tf.stack([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    L_image = tf.reshape(L_image, image_shape1)

    image_shape2 = tf.stack([IMAGE_HEIGHT, IMAGE_WIDTH, 2])
    AB_image = tf.reshape(AB_image, image_shape2)

    # image_shape3 = tf.stack([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    Mos_gray64 = tf.reshape(Mos_gray64, image_shape1)

    Mos_ab64 = tf.reshape(Mos_ab64, image_shape2)


    [L_image, AB_image, Mos_gray64, Mos_ab64] = tf.train.shuffle_batch([L_image, AB_image, Mos_gray64, Mos_ab64], batch_size=1, capacity=10, min_after_dequeue=1)

    return [L_image, AB_image, Mos_gray64, Mos_ab64]


''''# For testing function
def main():
    filename_queue = tf.train.string_input_producer([path_tfrecords_train], num_epochs=1)
    print(filename_queue)
    image = read_and_decode(filename_queue)
    # print(image)
    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        [L_image, AB_image, MOS_image_64, MOS_image_128, MOS_image_256, MOS_image_512, MOS_image_1024] = sess.run(image)
        print(L_image.shape)
        print(AB_image.shape)
        print(MOS_image_64.shape)
        print(MOS_image_128.shape)
        print(MOS_image_256.shape)
        print(MOS_image_512.shape)
        print(MOS_image_1024.shape)

        coord.request_stop()
        coord.join(threads)'''
