import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import vgg
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)
DEFAULT_PADDING = 'SAME'

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv'):
    with tf.variable_scope(scope):
        if pad > 0 :
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad

            else:
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)


        return x

def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape =[x_shape[0], x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_conneted(x, channels, use_bias=True, sn=False, scope='fully'):
    with tf.variable_scope(scope):
        x = tf.layers.flatten(x)
        shape = x.get_shape().as_list()
        x_channel = shape[-1]

        if sn :
            w = tf.get_variable("kernel", [x_channel, channels], tf.float32, initializer=weight_init, regularizer=weight_regularizer)
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else :
                x = tf.matmul(x, spectral_norm(w))

        else :
            x = tf.layers.dense(x, units=channels, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def gaussian_noise_layer(x, is_training=False):
    if is_training :
        noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
        return x + noise

    else :
        return x
    
def validate_padding(padding):
        assert padding in ('SAME', 'VALID')   
        
def max_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)
        
def make_var(name, shape, initializer=None, trainable=True):
    x = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return x
    
def conv_pretrain(input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
    validate_padding(padding)
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
#            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.1)
        init_weights = tf.truncated_normal_initializer(stddev=0.02)
        init_biases = tf.constant_initializer(0.0)
        kernel = make_var('weights', [k_h, k_w, int(c_i) / group, c_o], init_weights, trainable)
        biases = make_var('biases', [c_o], init_biases, trainable)
        if group == 1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        if relu:
            bias = tf.nn.bias_add(conv, biases)
            return tf.nn.relu(bias, name=scope.name)
        return tf.nn.bias_add(conv, biases, name=scope.name)
    
def fully_conneted_pretrain(input, num_in, num_out, name, Activation=None, padding=DEFAULT_PADDING, trainable=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.1)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [num_in,num_out], init_weights, trainable)
        biases = make_var('biases', [num_out], init_biases, trainable)        
        flat = tf.reshape(input, [-1, weights.get_shape().as_list()[0]])			
        act=tf.nn.xw_plus_b(flat,weights,biases,name=name)
        if Activation == 'relu':			
            relu=tf.nn.relu(act)			
            return relu		
        elif Activation == 'sigmoid':			
            sigmoid=tf.nn.sigmoid(act)			
            return sigmoid	  
        elif Activation == 'tanh':
            tanh=tf.nn.tanh(act)			
            return tanh	
        elif Activation == 'sofmax':
            sofmax=tf.nn.sofmax(act)
            return sofmax   
        elif Activation == None:
            return act

def DynFilter(x, F, filter_size):
    '''
    Dynamic filtering

    3D input x: (b, t, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (ft, fh, fw)

    input x: (b, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (fh, fw)

    '''
    # make tower
    filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)),
                                       (filter_size[1], filter_size[2], filter_size[0], np.prod(filter_size)))

    filter_localexpand = tf.Variable(filter_localexpand_np, trainable=False, dtype='float32',name='filter_localexpand')
    x_localexpand = tf.nn.conv2d(x, filter_localexpand, [1,1,1,1], 'SAME') # b, h, w, 1*5*5
    x_localexpand = tf.expand_dims(x_localexpand, axis=3)  # b, h, w, 1, 1*5*5
    x = tf.matmul(x_localexpand, F) # b, h, w, 1, R*R
    x = tf.squeeze(x, axis=3) # b, h, w, R*R

    return x

##################################################################################
# Unet
##################################################################################
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def Unet_conv(input, num_input_channels, conv_filter_size, num_filters, padding='SAME', relu=True):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding=padding)
    layer += biases

    if relu:
        layer = tf.nn.relu(layer)
    return layer

def pool_layer(input, padding='SAME'):
    return tf.nn.avg_pool(value=input,
                          ksize = [1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding=padding)

def Unet_deconv(input, num_input_channels, conv_filter_size, num_filters, feature_map_size_h, feature_map_size_w, padding='SAME',relu=True):

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_filters, num_input_channels])
    biases = create_biases(num_filters)
    batch_size_0 = 1
    layer = tf.nn.conv2d_transpose(value=input, filter=weights,
                                   output_shape=[batch_size_0, feature_map_size_h, feature_map_size_w, num_filters],
                                   strides=[1, 2, 2, 1],
                                   padding=padding)
    layer += biases

    if relu:
        layer = tf.nn.relu(layer)
    return layer

##################################################################################
# Block
##################################################################################

def resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = instance_norm(x)

        return x + x_init

def basic_block(x_init, channels, use_bias=True, sn=False, scope='basic_block') :
    with tf.variable_scope(scope) :
        x = lrelu(x_init, 0.2)
        x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        x = lrelu(x, 0.2)
        x = conv_avg(x, channels, use_bias=use_bias, sn=sn)

        shortcut = avg_conv(x_init, channels, use_bias=use_bias, sn=sn)

        return x + shortcut

def mis_resblock(x_init, z, channels, use_bias=True, sn=False, scope='mis_resblock') :
    with tf.variable_scope(scope) :
        z = tf.reshape(z, shape=[-1, 1, 1, z.shape[-1]])
        z = tf.tile(z, multiples=[1, x_init.shape[1], x_init.shape[2], 1]) # expand

        with tf.variable_scope('mis1') :
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, scope='conv3x3')
            x = instance_norm(x)

            x = tf.concat([x, z], axis=-1)
            x = conv(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_0')
            x = relu(x)

            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_1')
            x = relu(x)

        with tf.variable_scope('mis2') :
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, scope='conv3x3')
            x = instance_norm(x)

            x = tf.concat([x, z], axis=-1)
            x = conv(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_0')
            x = relu(x)

            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_1')
            x = relu(x)

        return x + x_init

def multidomain_resblock(x_init, y, z, channels, use_bias=True, sn=False, scope='multidomain_resblock') :
    with tf.variable_scope(scope) :
        y = tf.reshape(y, shape=[-1, 1, 1, y.shape[-1]])
        y = tf.tile(y, multiples=[1, x_init.shape[1], x_init.shape[2], 1])
        z = tf.reshape(z, shape=[-1, 1, 1, z.shape[-1]])
        z = tf.tile(z, multiples=[1, x_init.shape[1], x_init.shape[2], 1]) # expand

        with tf.variable_scope('mis1') :
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, scope='conv3x3')
            x = instance_norm(x)

            x = tf.concat([x, y, z], axis=-1)
            x = conv(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_0')
            x = relu(x)

            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_1')
            x = relu(x)

        with tf.variable_scope('mis2') :
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, scope='conv3x3')
            x = instance_norm(x)

            x = tf.concat([x, y, z], axis=-1)
            x = conv(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_0')
            x = relu(x)

            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_1')
            x = relu(x)

        return x + x_init
def avg_conv(x, channels, use_bias=True, sn=False, scope='avg_conv') :
    with tf.variable_scope(scope) :
        x = avg_pooling(x, kernel=2, stride=2)
        x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)

        return x

def conv_avg(x, channels, use_bias=True, sn=False, scope='conv_avg') :
    with tf.variable_scope(scope) :
        x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        x = avg_pooling(x, kernel=2, stride=2)

        return x

def expand_concat(x, z) :
    z = tf.reshape(z, shape=[z.shape[0], 1, 1, -1])
    z = tf.tile(z, multiples=[1, x.shape[1], x.shape[2], 1])  # expand
    x = tf.concat([x, z], axis=-1)

    return x

##################################################################################
# Sampling
##################################################################################

def down_sample(x) :
    return avg_pooling(x, kernel=3, stride=2, pad=1)

def avg_pooling(x, kernel=2, stride=2, pad=0) :
    if pad > 0 :
        if (kernel - stride) % 2 == 0:
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else:
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride, padding='VALID')

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    return gap

def z_sample(mean, logvar) :
    eps = tf.random_normal(shape=tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Load pretrain model
##################################################################################
def load_with_skip(data_path, session, skip_layer=None):
    data_dict = np.load(data_path, encoding='latin1', allow_pickle=True).item()
    for key in data_dict:
#        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))
                    print ("assign pretrain model " + subkey + " to " + key)

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(type, real, fake, fake_random=None, content=False):
    n_scale = len(real)
    loss = []

    real_loss = 0
    fake_loss = 0
    fake_random_loss = 0

    if content :
        for i in range(n_scale):
            if type == 'lsgan' :
                real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
                fake_loss = tf.reduce_mean(tf.square(fake[i]))

            if type =='gan' :
                real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

            loss.append(real_loss + fake_loss)

    else :
        for i in range(n_scale) :
            if type == 'lsgan' :
                real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
                fake_loss = tf.reduce_mean(tf.square(fake[i]))
                fake_random_loss = tf.reduce_mean(tf.square(fake_random[i]))

            if type == 'gan' :
                real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))
                fake_random_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_random[i]), logits=fake_random[i]))

            loss.append(real_loss * 2 + fake_loss + fake_random_loss)

    return sum(loss)


def generator_loss(type, fake, content=False):
    n_scale = len(fake)
    loss = []

    fake_loss = 0

    if content :
        for i in range(n_scale):
            if type =='lsgan' :
                fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 0.5))

            if type == 'gan' :
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=0.5 * tf.ones_like(fake[i]), logits=fake[i]))

            loss.append(fake_loss)
    else :
        for i in range(n_scale) :
            if type == 'lsgan' :
                fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))

            if type == 'gan' :
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))

            loss.append(fake_loss)


    return sum(loss)


def l2_regularize(x) :
    loss = tf.reduce_mean(tf.square(x))

    return loss

def kl_loss(mu, logvar) :
    loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)

    return loss

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def L2_loss(x, x_recon):
    loss = tf.losses.mean_squared_error(labels=x, predictions=x_recon)
    loss = tf.reduce_mean(loss)
    
    return loss

def softmax_cross_entropy(y_real, y_predict):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_real, logits=y_predict)
    loss = tf.reduce_mean(loss)
    
    return loss

def contrastive_loss(model1, model2, y, margin):
	distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
	similarity = y * tf.square(distance)                                           # keep the similar label (1) close to each other
	dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))        # give penalty to dissimilar label if the distance is bigger than margin

	return tf.reduce_mean(dissimilarity + similarity) / 2


def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op

    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)

    # 计算loss
    loss = tf.nn.l2_loss(features - centers_batch)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op

def centerloss(features, label, alfa, num_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers



def plot_confusion_matrix(cls_true, cls_pred, num_classes=8):    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    return plt.show()


##################################################################################
# Tsne function
##################################################################################

def plot_tsne_orign(source_img, source_label, target_img, target_label, samples=None, name="Samples_before_adaptation"):
    source_img = source_img.reshape((source_img.shape[0], -1))
    target_img = target_img.reshape((target_img.shape[0], -1))
    plot_tsne(source_img, source_label, target_img, target_label, samples, name)


def plot_tsne(source_feat, source_label, target_feat, target_label, samples=None, name="Samples_after_adaptation"):
    if samples != None:
        each_class_num = int(samples / 10)
        # source_feat = source_feat[:samples,:]
        # source_label = source_label[:samples]
        # target_feat = target_feat[:samples,:]
        # target_label = target_label[:samples]
        source_feat, source_label = select_samples(source_feat, source_label, each_class_num)
        target_feat, target_label = select_samples(target_feat, target_label, each_class_num)
        print(source_feat.shape)

    feat = np.concatenate([source_feat, target_feat], axis=0)
    tsne = TSNE(n_components=2, random_state=0)
    reduce_feat = tsne.fit_transform(feat)

    reduce_source_feat = reduce_feat[:source_feat.shape[0], :]
    reduce_target_feat = reduce_feat[source_feat.shape[0]:, :]
    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    plot_embedding(ax, reduce_source_feat, source_label, 0)
    plot_embedding(ax, reduce_target_feat, target_label, 1)
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend()
    plt.savefig("./result/" + name + ".jpg")


def select_samples(x, y, each_sample):
    ind = []
    count = [0] * 10
    for id, it in enumerate(y):
        if count[it] < each_sample:
            ind.append(id)
            count[it] += 1
        if sum(count) == each_sample * 10:
            break
    # print(ind)
    return x[ind], y[ind]


def plot_embedding(ax, X, d):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # plot color numbers
    if d == 0:
        label = "source"
    else:
        label = "target"
    ax.scatter(X[:, 0], X[:, 1], marker='.', color=plt.cm.bwr(d / 1.), label=label)






##################################################################################
# perceptual loss
##################################################################################
VGG_PATH = "/home/acm528_06/wei_zhong/DRIT_parkinglot/dataset/model/vgg19.mat"

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0],[-1, -1, -1, -1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

# TODO: Figure out grams and batch sizes! Doesn't make sense ..
def gram(layer):
    shape = tf.shape(layer)
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.stack([-1, num_filters]))
    gram = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(size)

    return gram
    

def get_style_features(style_images, style_layers):
    with tf.Session() as sess:
#        tf.reset_default_graph()
        net, _ = vgg.net(VGG_PATH, style_images)
        features = []
        for layer in style_layers:
            features.append(gram(net[layer]))
#    tf.reset_default_graph()
        with tf.Session() as sess:
            return sess.run(features)

def get_content_features(content_images, content_layers):
    with tf.Session() as sess:
        net, _ = vgg.net(VGG_PATH, content_images)
        layers = []
        for layer in content_layers:
            layers.append(net[layer])
        tf.reset_default_graph
        with tf.Session() as sess:
            return sess.run(layers + [content_images])

def compute_content_loss(content_features_t, content_layers, net):
    content_loss = 0
    for content_features, layer in zip(content_features_t, content_layers):
        layer_size = tf.size(content_features)
        content_loss += tf.nn.l2_loss(net[layer] - content_features) / tf.to_float(layer_size)
    content_loss = content_loss / len(content_layers)
    
    return content_loss


def compute_style_loss(style_features_t, style_layers, net):
    style_loss = 0
    for style_gram, layer in zip(style_features_t, style_layers):
        layer_size = tf.size(style_gram)
        style_loss += tf.nn.l2_loss(gram(net[layer]) - style_gram) / tf.to_float(layer_size)
        #style_loss += tf.sqrt(tf.reduce_sum(tf.pow(gram(net[layer]) - style_gram, 2)))
    style_loss =  style_loss / len(style_layers)
    
    return style_loss
