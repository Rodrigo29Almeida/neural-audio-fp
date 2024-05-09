# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" nnfp.py

'Neural Audio Fingerprint for High-specific Audio Retrieval based on 
Contrastive Learning', https://arxiv.org/abs/2010.11910

USAGE:
    
    Please see test() in the below.
    
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


# split de dados, pode criar um alyer que pode fazer um split de dados em 8 e colcar no forward e passar para a rede.
# criar a camada de split no init, posso nomear um layer que seja split. No functional crio uma funcao onde encadeio as camadas
# vai chegar a um ponto onde faço um split, criar um callback, posso nomear as funbçoes, dense_1 é um tf.keras.dese e tal, depois dense_2.
# depois rio uma funcao que encadeia esses nomes. no final x = split(x) depois forward/for (?) y em x. w=func paralelo, onde recebo o y do for(?)
# concatenate, depois é o retorn 

def FingerPrinter(input_shape=(256,32,1),
                 front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
                 front_strides=[[(1,2), (2,1)], [(1,2), (2,1)],
                                [(1,2), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)]],
                 emb_sz=128, # q
                 fc_unit_dim=[32,1],
                 norm='layer_norm2d',
                 use_L2layer=True):
    """
    Fingerprinter: 'Neural Audio Fingerprint for High-specific Audio Retrieval
        based on Contrastive Learning', https://arxiv.org/abs/2010.11910
    
    IN >> [Convlayer]x8 >> [DivEncLayer] >> [L2Normalizer] >> OUT 
    
    Arguments
    ---------
    input_shape: tuple (int), not including the batch size
    front_hidden_ch: (list)
    front_strides: (list)
    emb_sz: (int) default=128
    fc_unit_dim: (list) default=[32,1]
    norm: 'layer_norm1d' for normalization on Freq axis. 
          'layer_norm2d' for normalization on on FxT space (default).
          'batch_norm' or else, batch-normalization.
    use_L2layer: True (default)
    
    • Note: batch-normalization will not work properly with TPUs.
                    
    
    Input
    -----
    x: (B,F,T,1)
    
        
    Returns
    -------
    emb: (B,Q) 
    
    """
     
    input_layer = tf.keras.Input(shape=input_shape)


    # Front conv layers
    for i in range(len(front_strides)):
            front_conv = ConvLayer(hidden_ch=front_hidden_ch[i],
                strides=front_strides[i], norm=norm)
            
            conv = front_conv(input_layer)
    

    #podia fazer o flatten aqui?


    # Divide & Encoder layer
    div_enc_layer = DivEncLayer(q=emb_sz, unit_dim=fc_unit_dim, norm=norm)
    x = div_enc_layer(conv)

    if use_L2layer:
        x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        #return tf.math.l2_normalize(x, axis=1)
    
    return tf.keras.Model(inputs=input_layer, outputs=x)




def get_fingerprinter(conv_model, div_enc):
    fingerprinting_model = tf.keras.Sequential(name='Fingerprinting')
    fingerprinting_model.add(conv_model)
    fingerprinting_model.add(div_enc)
    fingerprinting_model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    
    """input_shape = (256, 32, 1)
    emb_sz = cfg['MODEL']['EMB_SZ']
    norm = cfg['MODEL']['BN']
    fc_unit_dim = [32, 1]
    
    m = FingerPrinter(input_shape=input_shape,
                      emb_sz=emb_sz,
                      fc_unit_dim=fc_unit_dim,
                      norm=norm)
    m.trainable = trainable"""
    return fingerprinting_model


conv_layer = FingerPrinter()
enc_layer = DivEncLayer()

finger_model = get_fingerprinting(conv_layer, enc_layer) 
    

def test():
    input_1s = tf.constant(np.random.randn(3,256,32,1), dtype=tf.float32) # BxFxTx1
    fprinter = FingerPrinter(emb_sz=128, fc_unit_dim=[32, 1], norm='layer_norm2d')
    emb_1s = fprinter(input_1s) # BxD
    
    input_2s = tf.constant(np.random.randn(3,256,63,1), dtype=tf.float32) # BxFxTx1
    fprinter = FingerPrinter(emb_sz=128, fc_unit_dim=[32, 1], norm='layer_norm2d')
    emb_2s = fprinter(input_2s)
    #%timeit -n 10 fprinter(_input) # 27.9ms
"""
Total params: 19,224,576
Trainable params: 19,224,576
Non-trainable params: 0

"""
