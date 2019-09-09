import tensorflow as tf
import numpy as np

tf.flags.DEFINE_string('name', None, 'Log directory')
flags = tf.flags.FLAGS


def tf_iterator():
    '''
    usage of tensorflow iterator
    :return:
    '''
    with tf.Session() as sess:
        data = [1, 2, 3, 4, 5, 6, 7]
        dataset = tf.data.Dataset.from_tensor_slices(data)
        batch = dataset.shuffle(4).batch(3)
        iterator = tf.data.Iterator.from_structure(batch.output_types,
                                                   batch.output_shapes)
        init = iterator.make_initializer(batch)
        # need initialize
        sess.run(init)
        print(sess.run(iterator.get_next()))
        print(sess.run(iterator.get_next()))
        print(sess.run(iterator.get_next()))


def tf_flags():
    '''
    usage of tensorflow flags
    :return:
    '''
    print(flags.name)


def tf_tfRecord_write(path, width, height, image_raw):
    '''
    write value to tf record
    :param path:
    :param width:
    :param height:
    :param image_raw:
    :return:
    '''
    test = [[1, 2, 3], [7, 7, 7]]
    test = np.array(test)
    c = test.astype(np.uint8)
    c_raw = c.tostring()


    # 有三种文件压缩格式可选，分别为TFRecordCompressionType.ZLIB、TFRecordCompressionType.GZIP以及TFRecordCompressionType.NONE，默认为最后一种，即不做任何压缩
    option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(path, options=None)

    # 第二步，tf.train.Feature生成协议信息
    feature_internal = {
        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "height": tf.train.Feature(float_list=tf.train.FloatList(value=[height])),
        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_raw]))
    }
    features_extern = tf.train.Features(feature=feature_internal)

    # 第三步，使用tf.train.Example将features编码数据封装成特定的PB协议格式
    example = tf.train.Example(features=features_extern)

    # 第四步，将example数据系列化为字符串
    example_str = example.SerializeToString()

    # 第五步，将系列化为字符串的example数据写入协议缓冲区
    writer.write(example_str)
    writer.flush()
    writer.close()


def tf_tfRead_read(filename):

    def _parse_function(record):
        # 定义一个特征词典，和写TFRecords时的特征词典相对应
        features = {
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.float32),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }

        # 根据上面的特征解析单个数据（保存时的单个Example）
        example = tf.parse_single_example(record, features)
        return example

    def _iterate_one(dataset):
        iterator = dataset.make_one_shot_iterator()
        result = iterator.get_next()
        width = result['width']
        height = result['height']
        img = result['image_raw']
        a, b, c = sess.run([width, height, img])
        print(a)
        print(b)
        print(np.array(list(c)))


    def _iterate_batch(dataset):
        batch = dataset.shuffle(2).batch(2)
        iterator = tf.data.Iterator.from_structure(batch.output_types,
                                                   batch.output_shapes)
        init = iterator.make_initializer(batch)
        sess.run(init)
        next = iterator.get_next()
        height = next['height']
        width = next['width']
        imgs = next['image_raw']
        a, b, c = sess.run([height, width, imgs])
        print(a)
        print(b)
        c_list = []
        for i in c:
            c_list.append(list(i))
        print(c_list)



    with tf.Session() as sess:
        # filename_queues = [filename]
        filename_queues = ['test.tfrecord', 'test1.tfrecord', 'test2.tfrecord']
        dataset = tf.data.TFRecordDataset(filename_queues)
        dataset = dataset.map(lambda x: _parse_function(x))
        _iterate_batch(dataset)







tf_tfRead_read('test.tfrecord')






