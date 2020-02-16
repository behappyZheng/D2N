import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np

class ImageData:

    def __init__(self, img_size, channels, augment_flag=False):
        self.img_size = img_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
#        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        x_decode = tf.image.decode_png(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            if self.img_size < 256 :
                augment_size = 256
            else :
                augment_size = self.img_size + 30
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img

########## parking lot data ##########	 
def Parkinglot_one_hot_encoding(target,lotclass):
    t = np.zeros((len(target),lotclass))
    for i in range(len(target)):
        index = target[i]
        t[i][int(index)] = 1
    return t
def load_parkinglotdata_from_file(images_file_path, label_file_path, start_sample, dataset_size, one_hot):
    file_img = open(images_file_path, 'rb')
    data = np.load(file_img)
    file_lab = open(label_file_path, 'rb')
    label = np.load(file_lab)
    end_sample = start_sample + dataset_size
    images = data[start_sample:end_sample]  
    labels = label[start_sample:end_sample]
    if one_hot == True:
        labels = Parkinglot_one_hot_encoding(labels ,8)
    return images, labels
def create_daytime_dataset(images_file_path, label_file_path, start_sample, dataset_size, batch_size, epochs, one_hot):  
    images, labels = load_parkinglotdata_from_file(images_file_path, label_file_path, start_sample, dataset_size, one_hot)  
    images = tf.cast(images, tf.float32) / 127.5 - 1 
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=len(str(dataset_size)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return dataset, iterator, next_element
def create_nighttime_dataset(images_file_path, label_file_path, start_sample, dataset_size, batch_size, epochs, one_hot):  
    images, labels = load_parkinglotdata_from_file(images_file_path, label_file_path, start_sample, dataset_size, one_hot)       
    images = tf.cast(images, tf.float32) / 127.5 - 1 
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    # dataset = dataset.shuffle(buffer_size=len(str(dataset_size)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return dataset, iterator, next_element
def create_nighttime_dataset_test(images_file_path, label_file_path, start_sample, dataset_size, batch_size, epochs, one_hot):
    images, labels = load_parkinglotdata_from_file(images_file_path, label_file_path, start_sample, dataset_size, one_hot)
    images = tf.cast(images, tf.float32) / 127.5 - 1 
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return dataset, iterator, next_element
def create_tsne_dataset(images_file_path, label_file_path, start_sample, dataset_size, one_hot):    
    images, labels = load_parkinglotdata_from_file(images_file_path, label_file_path, start_sample, dataset_size, one_hot)           
    images = tf.div(tf.subtract(images, tf.reduce_min(images)),
                            tf.subtract(tf.reduce_max(images), tf.reduce_min(images)))
    return images, labels


def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def augmentation(image, aug_img_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [aug_img_size, aug_img_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image
        img = misc.imresize(img, (144,96))

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def write_metadata(meta_path, tsne_point):
    f = open(meta_path, "w")
    f.truncate()
    f.write("Index\tLabel\n")
    for index in range(tsne_point):
        f.write("%d\t%d\n" % (index, 0))
    for index in range(tsne_point):
        f.write("%d\t%d\n" % (index + tsne_point, 1))

def write_metadata_iter(meta_path, tsne_point, batch):
    f = open(meta_path, "w")
    f.truncate()
    f.write("Index\tLabel\n")
    for index in range(0, int(tsne_point/batch)):
        for i in range(0, 2 * batch):
            if i < batch:
                f.write("%d\t%d\n" % (index * 2 * batch + i, 0))
            else:
                f.write("%d\t%d\n" % (index * 2 * batch + i, 1))