import os
import sys
import yaml

import tensorflow as tf
from tensorflow.keras.utils import Progbar
import tensorflow.keras as K
import librosa
import numpy as np
import glob
import pickle
import multiprocessing as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from model.dataset import Dataset
from model.fp.melspec.melspectrogram import get_melspec_layer
from model.fp.specaug_chain.specaug_chain import get_specaug_chain_layer
from model.fp.nnfp import get_fingerprinter

from librosa.util import frame



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def clean_gpu():
    from numba import cuda 
    device = cuda.get_current_device()
    device.reset()
clean_gpu()  # Reset GPU

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)



def build_fp(cfg):
    """ Build fingerprinter """
    # m_pre: log-power-Mel-spectrogram layer, S.
    m_pre = get_melspec_layer(cfg, trainable=False)

    # m_specaug: spec-augmentation layer.
    m_specaug = get_specaug_chain_layer(cfg, trainable=False)
    assert(m_specaug.bypass==False) # Detachable by setting m_specaug.bypass.

    # m_fp: fingerprinter g(f(.)).
    m_fp = get_fingerprinter(cfg, trainable=False)
    m_fp.trainable = False
    
    return m_pre, m_specaug, m_fp


def load_config(config_fname):
    config_filepath = './config/' + config_fname + '.yaml'
    if os.path.exists(config_filepath):
        print(f'cli: Configuration from {config_filepath}')
    else:
        sys.exit(f'cli: ERROR! Configuration file {config_filepath} is missing!!')

    with open(config_filepath, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


@tf.function
def predict(X, m_fp):
    """ 
    Test step used for mini-search-validation 
    X -> (B,1,8000)
    """
    emb_gf = m_fp(X)

    return emb_gf


def load_model():

    checkpoint_name_dir:str = "./logs/CHECKPOINT_BSZ_120"  #"CHECKPOINT"   # string
    config:str = "default"   

    cfg = load_config(config)

    _, _, m_fp = build_fp(cfg)

    checkpoint = tf.train.Checkpoint(m_fp)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_name_dir))
        
    return m_fp
    

def run(filepath, dstpath, m_fp):
    print(f"file entrada: {filepath}")
    signal, fs = librosa.load(filepath, mono=True, sr=8000)

    win_sz = fs
    hop_sz = int(fs/2)
        
    if len(signal) < win_sz:
        signal = librosa.util.pad_center(signal, size=win_sz, mode='constant')
        
    if len(signal) > 1.5*win_sz:
        frames = np.transpose(librosa.util.frame(signal, frame_length=win_sz, 
                                            hop_length=hop_sz))  # (B, 8000) # alterei de librosa.frame para librosa.util.frame
    else:
        frames = signal[:fs][None,:] #(1, 8000)


    #frames = frames[..., np.newaxis] #(B,8000,1) -- Adicionar mais uma dimensão aos dados 
    
    X = frames[np.newaxis, np.newaxis, ...]  #(1,B,8000,1)  
    X = tf.convert_to_tensor(X, dtype=tf.float32)  # (1,B,8000,1)
    X = tf.transpose(X, perm=[2, 0, 1, 3]) # (B,1,1,8000)
    
    emb = predict(X, m_fp)

    tf.config.run_functions_eagerly(True)

    print(f"Saving features to: {dstpath}")
    with open(dstpath, "wb") as f:
        pickle.dump(emb.numpy(), f)


    
if __name__ == "__main__":
        
    files_dummy_db_dir = '/mnt/dataset/public/Fingerprinting/neural-audio-fp-dataset/music/test-dummy-db-100k-full/fma_full'
    #files_dummy_db_dir = '/mnt/dataset/test/audio'
    files_query_dir = '/mnt/dataset/public/Fingerprinting/neural-audio-fp-dataset/music/test-query-db-500-30s'
    root_out = '/mnt/dataset/public/Fingerprinting/features'
    
    files = glob.glob(os.path.join(files_dummy_db_dir, "**/*.wav")) + glob.glob(os.path.join(files_query_dir, "**/*.wav"))
    


    parallel = False
    model_fp = load_model()

    if parallel:

        pool = mp.Pool()  #num_proc)
        print(f"pool:{pool}")

        for src in files:
            
            parts = src.split("/")
            subdir= parts[-3]
            set_id = parts[-2]
            track = parts[-1].split(".")[0]

            out_path = os.path.join(root_out, subdir, set_id)
            os.makedirs(out_path, exist_ok=True)

            dst = os.path.join(out_path, track + ".pkl")
            
            if not os.path.exists(dst):
                #print(dst)
                pool.apply_async(run, (src, dst, model_fp))

        pool.close()
        pool.join()

    else:
        
        for src in files:
            parts = src.split("/")
            subdir= parts[-3]
            set_id = parts[-2]
            track = parts[-1].split(".")[0]

            out_path = os.path.join(root_out, subdir, set_id)
            os.makedirs(out_path, exist_ok=True)

            dst = os.path.join(out_path, track + ".pkl")

            if not os.path.exists(dst):
                #print(dst)
                run(src, dst, model_fp)

    clean_gpu()