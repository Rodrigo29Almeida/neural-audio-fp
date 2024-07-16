# -*- coding: utf-8 -*-
"""melsprctrogram.py"""    
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda, Permute
from kapre.time_frequency import STFT, ApplyFilterbank
from tensorflow.keras.layers import Layer
import math

import numpy as np


from kapre import backend
from kapre.backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR
from tensorflow.keras import backend as K


class CustomApplyFilterbank(ApplyFilterbank):
    def __init__(self, type, filterbank_kwargs, data_format='default', **kwargs):
        super(CustomApplyFilterbank, self).__init__(type, filterbank_kwargs, data_format, **kwargs)

        # tipo de Banco de filtros
        if type == 'tri':
            self.filterbank = self.filterbank_triangular_log(**filterbank_kwargs)

        if data_format == _CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        if self.data_format == _CH_FIRST_STR:
            self.freq_axis = 3
        else:
            self.freq_axis = 2

    def filterbank_triangular_log(self, sample_rate, n_fft):
        #este texo abaixo está
        # Com o objetivo de ter 256 filtros e 8000 Hz na frequência de amostragem, teve-se de optar por Nfft de 2048, o que resulta em 54.4024 filtros por oitava, e numa frequência mínima de 151.3483 Hz.

        # Sendo assim, o Nfpo será 60, 5*12, e a frequência do último filtro, f256, será Si7 = 3951.066410048992 Hz. Resultando numa frequência máxima de 3996.975590329487 Hz.
        # Com isto, obtem-se pelo menos um bin em cada filtro, visto que f0*(2^(2/Nfpo)-1) = 4.7979 > 8000/2048 = 3.9062. Para uma Nfft de 1024, não era certo que obtivesse pelo menos um bin por filtro.
        #O primeiro filtro estava a zero.

        n_fft=2048
        sample_rate=8000
        Nfpo=60 #=5*12
        Nb =256

        #Cálculo da fmin e fmax
        f256 = 440*2.**(38/12) # Si7 = 3951.066410048992 Hz;
        f0=f256/2**(256/Nfpo) # fmin, 205.2672581380976 Hz
        fmax = f0*2**(257/Nfpo) # fmax, 3996.975590329487 Hz

        #Depois disto, dá bins em todos os fitros. Ver a linha 24 do getOctaveFilterBanck2.m

        i=np.arange(0,Nb+2, dtype=np.int32)
        k=np.arange(0, n_fft//2+1, dtype=int)
        f=k*sample_rate/n_fft

        fcf = f0 * 2.**(i/Nfpo) #3905.68454168

        #fi = np.concatenate(([f0], fcf, [fmax])) #fi =[f0, fcf, fmax] 
        #fcf está a incluir f0, fcf e fmax

        # Construct the output matrix
        H = np.zeros((Nb, n_fft // 2 + 1))

        #for i in range(n_filters), com isto são 256
        for j in range(Nb):
            fLow = fcf[j] 
            fmid = fcf[j+1] 
            fUpp = fcf[j+2]

            H[j, :] = ((f - fLow) / (fmid - fLow)) * ((f > fLow) & (f <= fmid)) + \
                            ((f - fUpp) / (fmid - fUpp)) * ((f > fmid) & (f <= fUpp))
            

        return tf.convert_to_tensor(H.T, dtype=tf.float32)
        #return tf.convert_to_tensor(H.T)
    

    def call(self, x):
        output = tf.tensordot(x, self.filterbank, axes=(self.freq_axis, 0))
                
        if self.data_format == _CH_LAST_STR:
            output = tf.transpose(output, (0, 1, 3, 2))
        return output



class MagnitudeSquared(Layer):
    def call(self, x):
        return tf.abs(x)**2


class Melspec_layer(Model):
    """
    A wrapper class, based on the implementation:
        https://github.com/keunwoochoi/kapre
        
    Input:
        (B,1,T)
    Output:
        (B,C,T,1) with C=Number of mel-bins
    
    USAGE:
        
        See get_melspec_layer() in the below.
        
    """
    def __init__(
            self,
            input_shape=(1, 8000),
            segment_norm=False,
            n_fft=2048,
            stft_hop=258,
            n_win=1549, # = 6*M+1, Window Length
            n_mels=256,
            fs=8000,
            dur=1.,
            f_min=300,
            f_max=4000.,
            amin=10**(-20/10), # minimum amp.
            dynamic_range=80.,
            use_pad_layer=False,
            name='Mel-spectrogram',
            trainable=False,
            **kwargs
            ):
        super(Melspec_layer, self).__init__(name=name, trainable=False, **kwargs)
        
        self.mel_fb_kwargs = {
            'sample_rate': fs,
            'n_freq': n_fft // 2 + 1,
            'n_mels': n_mels,
            'f_min': f_min,
            'f_max': f_max,
            }
        
        self.tri_fb_kwargs = {
            'sample_rate': fs,
            'n_fft': n_fft,
            }
            
        self.n_fft = n_fft
        self.stft_hop = stft_hop
        self.n_mels = n_mels
        self.amin = amin
        self.dynamic_range = dynamic_range
        self.segment_norm = segment_norm

        self.n_win = n_win #RA

        #Padding para type ='mel'
         # 'SAME' Padding layer
        self.use_pad_layer = use_pad_layer
        self.pad_l = n_fft // 2
        self.pad_r = n_fft // 2
        self.padded_input_shape = (1, int(fs * dur) + self.pad_l + self.pad_r)
        self.pad_layer = Lambda(
            lambda z: tf.pad(z, tf.constant([[0, 0], [0, 0],
                                             [self.pad_l, self.pad_r]]))
            )

        #Padding para type ='tri'
        self.Npad_l = 3*self.stft_hop
        self.Npad_r = 3*self.stft_hop-1
        self.padded_input_shape_RA = (1, int(fs * dur) + self.Npad_l + self.Npad_r) #self.padded_input_shape:(1, 9547)
        self.pad_layer_RA = Lambda(
            lambda z: tf.pad(z, tf.constant([[0, 0], [0, 0],
                                             [self.Npad_l, self.Npad_r]]))
            )  #onde fs=31*M+2
        

        # Construct log-power Mel-spec layer
        self.m = self.construct_melspec_layer(input_shape, name)


        # Permute layer
        self.p = tf.keras.Sequential(name='Permute')
        self.p.add(Permute((3, 2, 1), input_shape=self.m.output_shape[1:]))
        
        super(Melspec_layer, self).build((None, input_shape[0], input_shape[1]))
        
        
    def construct_melspec_layer(self, input_shape, name):
        m = tf.keras.Sequential(name=name)
        m.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        m.add(self.pad_layer_RA)#RA - podia ser um if para usar o outro pad
        m.add(
            STFT(
                n_fft=self.n_fft,
                win_length=self.n_win,
                hop_length=self.stft_hop,
                window_name='hamming_window',
                pad_begin=False, # We do not use Kapre's padding, due to the @tf.function compatiability
                pad_end=False, # We do not use Kapre's padding, due to the @tf.function compatiability
                input_data_format='channels_first',
                output_data_format='channels_first')
            )
        m.add(
            MagnitudeSquared()
            )
        m.add(
            CustomApplyFilterbank(type='tri', #pode ser 'tri', 'log', ou 'mel'
                            filterbank_kwargs=self.tri_fb_kwargs,
                            data_format='channels_first'
                            )
            )
        return m
        

    @tf.function
    def call(self, x):
        x = self.m(x) + 0.06
        #x = tf.sqrt(x)
        
        x = tf.math.log(tf.maximum(x, self.amin)) / math.log(10) # ver se usa o 20 aqui, com 20 ficava em db, sem 20 não
        #x = 10*(tf.math.log(tf.maximum(x, self.amin)) / math.log(10)) # ver se usa o 20 aqui, com 20 ficava em db, sem 20 não
        x = x - tf.reduce_max(x)
        x = tf.maximum(x, -1 * self.dynamic_range) #ver os dados do x aqui
        if self.segment_norm:
            x = (x - tf.reduce_min(x) / 2) / tf.abs(tf.reduce_min(x) / 2 + 1e-10)
        return self.p(x) # Permute((3,2,1))
    

def get_melspec_layer(cfg, trainable=False):
    #type - 'mel' filter bank
    """fs = cfg['MODEL']['FS']
    dur = cfg['MODEL']['DUR']
    n_fft = cfg['MODEL']['STFT_WIN']
    stft_hop = cfg['MODEL']['STFT_HOP']
    n_mels = cfg['MODEL']['N_MELS']
    f_min = cfg['MODEL']['F_MIN']
    f_max = cfg['MODEL']['F_MAX']
    if cfg['MODEL']['FEAT'] == 'melspec':
        segment_norm = False
    elif cfg['MODEL']['FEAT'] == 'melspec_maxnorm':
        segment_norm = True
    else:
        raise NotImplementedError(cfg['MODEL']['FEAT'])
    use_pad_layer = True"""

    #type - 'tri' filter bank
    fs = cfg['MODEL']['FS']
    dur = cfg['MODEL']['DUR']
    n_fft = cfg['MODEL']['STFT_WIN']
    stft_hop = cfg['MODEL']['STFT_HOP']
    n_win = cfg['MODEL']['N_WIN'] #RA
    n_mels = cfg['MODEL']['N_MELS']
    f_min = cfg['MODEL']['F_MIN']
    f_max = cfg['MODEL']['F_MAX']
    if cfg['MODEL']['FEAT'] == 'melspec':
        segment_norm = False
    elif cfg['MODEL']['FEAT'] == 'melspec_maxnorm':
        segment_norm = True
    else:
        raise NotImplementedError(cfg['MODEL']['FEAT'])
    use_pad_layer = False
    
    input_shape = (1, int(fs * dur))
    l = Melspec_layer(input_shape=input_shape,
                      segment_norm=segment_norm,
                      n_fft=n_fft,
                      stft_hop=stft_hop,
                      n_win=n_win, #RA
                      n_mels=n_mels,
                      fs=fs,
                      dur=dur,
                      f_min=f_min,
                      f_max=f_max,
                      use_pad_layer=use_pad_layer)#RA
    l.trainable = trainable
    return l
                        