import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input

def convlayer(hidden_ch=128,
              strides=[(1,1),(1,1)],
              norm='layer_norm2d'):
    conv2d_1x3 = tf.keras.layers.Conv2D(hidden_ch,
                                        kernel_size=(1, 3),
                                        strides=strides[0],
                                        padding='SAME',
                                        dilation_rate=(1, 1),
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros')
    conv2d_3x1 = tf.keras.layers.Conv2D(hidden_ch,
                                        kernel_size=(3, 1),
                                        strides=strides[1],
                                        padding='SAME',
                                        dilation_rate=(1, 1),
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros')
    if norm == 'layer_norm1d':
        BN_1x3 = tf.keras.layers.LayerNormalization(axis=-1)
        BN_3x1 = tf.keras.layers.LayerNormalization(axis=-1)
    elif norm == 'layer_norm2d':
        BN_1x3 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
        BN_3x1 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
    else:
        BN_1x3 = tf.keras.layers.BatchNormalization(axis=-1)
        BN_3x1 = tf.keras.layers.BatchNormalization(axis=-1)
        
    forward = tf.keras.Sequential([conv2d_1x3,
                                   tf.keras.layers.ELU(),
                                   BN_1x3,
                                   conv2d_3x1,
                                   tf.keras.layers.ELU(),
                                   BN_3x1
                                   ])
    
    return forward


def create_sequential_front_conv(input_shape=(256,32,1),
                                 emb_sz=128,
                                 front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
                                 front_strides=[[(1,2), (2,1)], [(1,2), (2,1)],
                                                [(1,2), (2,1)], [(1,2), (2,1)],
                                                [(1,1), (2,1)], [(1,2), (2,1)],
                                                [(1,1), (2,1)], [(1,2), (2,1)]],
                                 norm='layer_norm2d'):
    front_conv = tf.keras.Sequential(name='ConvLayers')
    if ((front_hidden_ch[-1] % emb_sz) != 0):
        front_hidden_ch[-1] = ((front_hidden_ch[-1]//emb_sz) + 1) * emb_sz

    for i in range(len(front_strides)):
        front_conv.add(convlayer(hidden_ch=front_hidden_ch[i], strides=front_strides[i], norm=norm))
    front_conv.add(tf.keras.layers.Flatten())

    return front_conv


def auxiliar(input1):
    conv_layer = create_sequential_front_conv(input_shape=(256,32,1),
                                               emb_sz=128,
                                               front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
                                               front_strides=[[(1,2), (2,1)], [(1,2), (2,1)],
                                                              [(1,2), (2,1)], [(1,2), (2,1)],
                                                              [(1,1), (2,1)], [(1,2), (2,1)],
                                                              [(1,1), (2,1)], [(1,2), (2,1)]],
                                               norm='layer_norm2d')

    unit_dim = [32, 1]
    q = 128
    arquiteturas_densas = tf.keras.Sequential([tf.keras.layers.Dense(unit_dim[0], activation='elu'),
                                               tf.keras.layers.Dense(unit_dim[1])])

    x = input1
    #x reshape
    x = conv_layer(x)

    y_list = [0] * q
    x_split = tf.split(x, num_or_size_splits=128, axis=1)

    for v, k in enumerate(x_split):
        y_list[v] = arquiteturas_densas(k)

    out = tf.concat(y_list, axis=1)
    output = tf.math.l2_normalize(out, axis=1)
    return output


def get_fingerprinting(input1):
    output = auxiliar(input1)
    fingerprinting_model = Model(inputs=input1, outputs=output)
    return fingerprinting_model


"""



import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input

def convlayer(hidden_ch=128,
              strides=[(1,1),(1,1)],
              norm='layer_norm2d'):
    
    
    conv2d_1x3 = tf.keras.layers.Conv2D(hidden_ch,
                                        kernel_size=(1, 3),
                                        strides=strides[0],
                                        padding='SAME',
                                        dilation_rate=(1, 1),
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros')
    conv2d_3x1 = tf.keras.layers.Conv2D(hidden_ch,
                                        kernel_size=(3, 1),
                                        strides=strides[1],
                                        padding='SAME',
                                        dilation_rate=(1, 1),
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros')
    
    if norm == 'layer_norm1d':
        BN_1x3 = tf.keras.layers.LayerNormalization(axis=-1)
        BN_3x1 = tf.keras.layers.LayerNormalization(axis=-1)
    elif norm == 'layer_norm2d':
        BN_1x3 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
        BN_3x1 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
    else:
        BN_1x3 = tf.keras.layers.BatchNormalization(axis=-1) # Fix axis: 2020 Apr20
        BN_3x1 = tf.keras.layers.BatchNormalization(axis=-1)
        
    forward = tf.keras.Sequential([conv2d_1x3,
                                    tf.keras.layers.ELU(),
                                    BN_1x3,
                                    conv2d_3x1,
                                    tf.keras.layers.ELU(),
                                    BN_3x1
                                    ])
    
    return forward


def create_sequential_front_conv(input_shape=(256,32,1),
                                    emb_sz=128, # q
                                    front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
                                    front_strides=[[(1,2), (2,1)], [(1,2), (2,1)],
                                                   [(1,2), (2,1)], [(1,2), (2,1)],
                                                   [(1,1), (2,1)], [(1,2), (2,1)],
                                                   [(1,1), (2,1)], [(1,2), (2,1)]],
                                    norm='layer_norm2d'):

    front_conv = tf.keras.Sequential(name='ConvLayers')
    if ((front_hidden_ch[-1] % emb_sz) != 0):
        front_hidden_ch[-1] = ((front_hidden_ch[-1]//emb_sz) + 1) * emb_sz

    for i in range(len(front_strides)):
        front_conv.add(convlayer(hidden_ch=front_hidden_ch[i], strides=front_strides[i], norm=norm))
    front_conv.add(tf.keras.layers.Flatten())

    return front_conv



def auxiliar(input1):
    conv_layer = create_sequential_front_conv(input_shape=(256,32,1),
                                        emb_sz=128, # q
                                        front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
                                        front_strides=[[(1,2), (2,1)], [(1,2), (2,1)],
                                                    [(1,2), (2,1)], [(1,2), (2,1)],
                                                    [(1,1), (2,1)], [(1,2), (2,1)],
                                                    [(1,1), (2,1)], [(1,2), (2,1)]],
                                        norm='layer_norm2d')

    unit_dim=[32, 1]
    q=128
    arquiteturas_densas = tf.keras.Sequential([tf.keras.layers.Dense(unit_dim[0], activation='elu'),
                                            tf.keras.layers.Dense(unit_dim[1])])


    x=input1
    #x = tf.reshape(x, shape=[x.shape[0], q, -1])
    x = conv_layer(x)
    #x = tf.reshape(x, shape=[x.shape[0], q, -1])

    y_list = [0]*q # lista vazia de tamanho 128, para guardar as 128 redes


    x_split = tf.split(x, num_or_size_splits=128, axis=1) # faz o split em 128 vetores de igual tamanho (8)

    #Dados as 128 redes
    for v, k in enumerate(x_split):
        y_list[v] = arquiteturas_densas(k)


    out = tf.concat(y_list, axis=1)
    output = tf.math.l2_normalize(out, axis=1)
    return output



def get_fingerprinting(input1):
    
    fingerprinting_model = tf.keras.Sequential(name='Fingerprinting')
    output = auxiliar(input1)
    fingerprinting_model = Model(inputs=input1, outputs=output)
    
    return fingerprinting_model

"""


"""
import numpy as np
import tensorflow as tf
assert tf.__version__ >= "2.0"

def ConvLayer(hidden_ch=128,
                 strides=[(1,1),(1,1)],
                 norm='layer_norm2d'):
    
    #input_layer = tf.keras.Input(shape=(None, None, 1))
    
    #Convolution 1x3
    conv2d_1x3 = tf.keras.layers.Conv2D(hidden_ch,
                                                 kernel_size=(1, 3),
                                                 strides=strides[0],
                                                 padding='SAME',
                                                 dilation_rate=(1, 1),
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='zeros')(input_layer)

    layerNorm_1x3 = tf.keras.layers.ELU()(conv2d_1x3)

    if norm == 'layer_norm1d':
        BN_1x3 = tf.keras.layers.LayerNormalization(axis=-1)(layerNorm_1x3)
    elif norm == 'layer_norm2d':
        BN_1x3 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))(layerNorm_1x3)
    else:
        BN_1x3 = tf.keras.layers.BatchNormalization(axis=-1)(layerNorm_1x3) # Fix axis: 2020 Apr20


    #Convolution 3x1
    conv2d_3x1 = tf.keras.layers.Conv2D(hidden_ch,
                                                 kernel_size=(3, 1),
                                                 strides=strides[1],
                                                 padding='SAME',
                                                 dilation_rate=(1, 1),
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='zeros')(BN_1x3)

    layerNorm_3x1 = tf.keras.layers.ELU()(conv2d_3x1)

    if norm == 'layer_norm1d':
        BN_3x1 = tf.keras.layers.LayerNormalization(axis=-1)(layerNorm_3x1)
    elif norm == 'layer_norm2d':
        BN_3x1 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))(layerNorm_3x1)
    else:
        BN_3x1 = tf.keras.layers.BatchNormalization(axis=-1)(layerNorm_3x1)

    
    return tf.keras.Model(inputs=input_layer, outputs=BN_3x1)



def DivEncLayer(q=128, unit_dim=[32, 1], norm='batch_norm'):

    input_layer = tf.keras.Input(shape=(None, None, 1))
    flatten_layer = tf.keras.layers.Flatten()(input_layer)

    layers = []

    for i in range(q):
        layers.dense_1 = tf.keras.layers.Dense(unit_dim[0], activation='elu')(flatten_layer)
        layers.dense_2 = tf.keras.layers.Dense(unit_dim[1])(layers.dense_1)

    return tf.keras.Model(inputs=input_layer, outputs=tf.keras.layers.Concatenate(axis=1)(layers))
"""


# split de dados, pode criar um alyer que pode fazer um split de dados em 8 e colcar no forward e passar para a rede.
# criar a camada de split no init, posso nomear um layer que seja split. No functional crio uma funcao onde encadeio as camadas
# vai chegar a um ponto onde faço um split, criar um callback, posso nomear as funbçoes, dense_1 é um tf.keras.dese e tal, depois dense_2.
# depois rio uma funcao que encadeia esses nomes. no final x = split(x) depois forward/for (?) y em x. w=func paralelo, onde recebo o y do for(?)
# concatenate, depois é o retorn 