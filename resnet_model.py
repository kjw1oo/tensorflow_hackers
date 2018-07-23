
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot = True)
print('wow')
#residual learning
# f(x) + x
# original resnet : 본래 분기되기전에 활성화 함수 적용 
# this resnet : 분기된 이후 shortcut은 그냥 냅두고 weight path의 weight layer 앞에 활성화함수를 두는 pre_activation 설정
# regularization 효과

def residual_unit(inputs, n_filter,  strides, name, short_cut = True, is_training = True, reuse = False):
    # 쇼트컷 (residual) 설정
    shortcut = inputs
    if short_cut:
        shortcut = tf.layers.conv2d(inputs = shortcut,
                                    filters = n_filter,
                                    kernel_size= 1,
                                    strides = strides,
                                    padding = 'SAME',
                                    name = name + 'shortcut',
                                    reuse =reuse)
        
    #pre_activation
    L = tf.layers.batch_normalization(inputs, training = is_training, name = name + 'BN1', reuse = reuse)
    L = tf.nn.relu(L)
    # convolution (weight layer)
    L = tf.layers.conv2d(inputs = L,
                         filters = n_filter,
                         kernel_size = 3,
                         strides = strides,
                         padding = 'SAME',
                         name = name + 'conv1',
                         reuse = reuse)
    L = tf.layers.batch_normalization(L, training = is_training, name = name + 'BN2', reuse = reuse)
    L = tf.nn.relu(L)
    # 블럭의 마지막 layer는 언제나 strides = 1로 하여야한다.
    L = tf.layers.conv2d(inputs = L,
                         filters = n_filter,
                         kernel_size = 3,
                         strides = 1,
                         padding = 'SAME',
                         name = name + 'conv2',
                         reuse = reuse)
    
    return L + shortcut
    
 # 학습에 걸리는 시간을 고려해서 50 layer 이상에서는 bottle_neck 구조를 사용한다.
# bottle_neck구조는 output filter의 갯수가 4배로 증가되어 병부리처럼 보여서 bottle_neck이라고 한다.
def bottle_neck_unit(inputs, n_filter, strides, name, short_cut = True, is_training = True, reuse = False):
    shortcut = inputs
    # bottle_neck 레이어에서는 output filter의 갯수가 input filter 갯수의 4배
    n_filter_out = n_filter
    if short_cut:
        shortcut = tf.layers.conv2d(inputs = shortcut,
                                    filters = n_filter_out,
                                    kernel_size= 1,
                                    strides = strides,
                                    padding = 'SAME',
                                    name = name + 'shortcut',
                                    reuse =reuse)
        
    # pre_activation
    L =  tf.layers.batch_normalization(inputs, training = is_training, name = name + 'BN1', reuse = reuse)
    L = tf.nn.relu(L)
    # 첫번째 레이어 conv 1x1
    L = tf.layers.conv2d(inputs = L,
                         filters = n_filter,
                         kernel_size = 1,
                         strides = 1,
                         padding = 'SAME',
                         name = name + 'conv1',
                         reuse=reuse)
    
    L =  tf.layers.batch_normalization(inputs, training = is_training, name = name + 'BN2', reuse = reuse)
    L = tf.nn.relu(L)
    # 두번쨰 레이어 conv 3x3, strides = strides
    L = tf.layers.conv2d(inputs = L,
                         filters = n_filter,
                         kernel_size = 3,
                         strides = strides,
                         padding = 'SAME',
                         name = name + 'conv2',
                         reuse=reuse)
    L =  tf.layers.batch_normalization(inputs, training = is_training, name = name + 'BN3', reuse = reuse)
    L = tf.nn.relu(L)
    # 세번째 레이어 conv 1x1, n_filter x 4
    L = tf.layers.conv2d(inputs = L,
                         filters = n_filter_out,
                         kernel_size = 1,
                         strides = 1,
                         padding = 'SAME',
                         name = name + 'conv3',
                         reuse=reuse)
    
    return L + shortcut

# 1. params 에서 각 블럭레이어에서 어떤 블럭을 몇개 쌓을 것인지 결정한다.( 34 layer기준 3, 4, 6, 3 residual_net)
# 2. 첫번째 블럭은 strides를 2로 설정하여 이미지 크기를 줄인다.
# 3. 나머지 블럭은 strides를 1로 설정.
def block_layer(inputs, n_filter, strides, name, n_block, is_bottleneck = False, is_training = True, reuse = False):
    # bottleneck_layer를 사용한다면
    if is_bottleneck:
        # 첫번째 블럭은 strides 파라미터 받은데로 설정(이미지 크기 1/2)
        L = bottle_neck_unit(inputs, n_filter, strides, name+'_bottle_neck_1_', is_training = is_training, reuse = reuse)
        # 나머지 블럭은 strides를 1로 설정(이미지 크기 보존)
        for i in range(2, n_block + 1):
            L = bottle_neck_unit(inputs, n_filter, 1, name+'_bottle_neck_'+ str(i) + '_', is_training = is_training, reuse = reuse)
    else:
        # 첫번째 블럭은 strides 파라미터 받은데로 설정(이미지 크기 1/2)
        L = residual_unit(inputs, n_filter, strides, name+'_residual_unit_1_', is_training = is_training, reuse = reuse)
        # 나머지 블럭은 strides를 1로 설정(이미지 크기 보존)
        for i in range(2, n_block + 1):
            L = residual_unit(inputs, n_filter, 1, name+'_residual_unit_'+ str(i) + '_', is_training = is_training, reuse = reuse)
            
    return L
def get_logits(inputs, resnet_params, is_training = True, reuse = False):
    
    start_n_filters = resnet_params['start_n_filters']
    start_strides = resnet_params['start_strides']
    n_filters = resnet_params['n_filters']
    n_blocks = resnet_params['n_blocks']
    is_bottlenecks = resnet_params['is_bottlenecks']
    layers_strides = resnet_params['layers_strides']
    
    layer_number = range(1, len(n_blocks) + 1)
    
    
    # 첫번째는 먼저 conv 3x3 시행 128x32 -> 64 x 16 
    L = tf.layers.conv2d(inputs = inputs,
                         filters = start_n_filters,
                         kernel_size = 3,
                         strides = start_strides,
                         padding = 'SAME',
                         name = 'begin_conv',
                         reuse = reuse)
    # 크기가 큰경우는 max_pooling을 거치나 여기는 그렇게 크지않으므로 제외
    # strides = [1, 2, 2] 기준
    # 첫번째 layer image size : 64 x16 -> 64 x 16
    # 두번째 layer image size : 64 x 16 -> 32 x 8
    # 세번째 layer image size : 32 x 8 -> 16 x 4
    for i, n_filter, strides, n_block, is_bottleneck in zip(layer_number, n_filters, layers_strides, n_blocks, is_bottlenecks):
        L = block_layer(inputs = L,
                        n_filter = n_filter,
                        strides = strides,
                        name = 'block'+str(i),
                        n_block = n_block,
                        is_bottleneck = is_bottleneck,
                        reuse = reuse)
        
    #마지막 batch_norm and activation function
    outputs =  tf.layers.batch_normalization(L, training = is_training, name='final_BN', reuse = reuse)
    outputs = tf.nn.relu(L)
    # global average_pooling
    # shape : (batch_size, height, width, n_feature_map)
    shape = outputs.get_shape().as_list()
    # 글로벌 풀링 사이즈 (height, width)
    pool_size = (shape[1], shape[2])
    outputs= tf.layers.average_pooling2d(outputs, pool_size = pool_size, strides = 1, padding = 'VALID')
    # 마지막 dense layer
    outputs = tf.layers.flatten(outputs)
    outputs = tf.layers.dense(outputs, 10, name = 'final_dense', reuse=reuse)
    return outputs

class resnet:
    def __init__(self, sess, params, model_name):
        self.sess = sess
        self.batch_size = 50
        self.params = params
        self.model_name = model_name
        self._build_net()
        
        
    def _build_net(self):
        learning_rate= self.params['learning_rate']
        batch_size= self.params['batch_size']
        epochs= self.params['epochs']
        height= self.params['height']
        width= self.params['width']
        model_path= self.params['model_path']
        
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        with tf.variable_scope(self.model_name):
            self.logits_train = get_logits(self.X, self.params)                              
            self.loss = tf.losses.softmax_cross_entropy(self.Y, self.logits_train)   

            self.logits_eval = get_logits(self.X, self.params, is_training = False, reuse = True)
            self.predict_proba_ = tf.nn.softmax(self.logits_eval)
            self.prediction = tf.argmax(self.predict_proba_, 1)
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.accuracy = tf.metrics.accuracy(tf.argmax(self.Y, 1), self.prediction)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_name)            
        with tf.control_dependencies(update_ops):    
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables())

        #accuracy = tf.metrics.accuracy(tf.argmax(Y, 1), prediction)
        # 변수들 프린트/ 텐서보드 summary 생성
        for i, v in enumerate(tf.trainable_variables()):
            print('number : {} )) {}'.format(i, v))
            
    def fit(self):
        total_batch = int(mnist.train.num_examples/ self.batch_size)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        for epoch in range(10):
            total_cost = 0
        
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
                _, c = sess.run([self.optimizer, self.loss], feed_dict = {self.X:batch_xs.reshape(-1, 28, 28, 1), self.Y: batch_ys})
            acc = sess.run(self.accuracy, feed_dict = {self.X: mnist.test.images.reshape(-1, 28, 28, 1), self.Y: mnist.test.labels})
            
            print('epoch : {}, cost : {}, acc: {}'.format(epoch, c, acc))
        self.saver.save(self.sess, self.model_path+self.name+'.ckpt', global_step = self.sess.run(self.global_step))
        return self


# In[ ]:



