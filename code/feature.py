

import os
from glob2 import glob
import numpy as np
from python_speech_features import delta,fbank
import librosa
import joblib
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

feature_path = '../user_data/feature'

seed = 2020
np.random.seed(seed)

win_len = 0.02 * 300
step_len =  win_len/2

winlen=0.025
winstep=0.01


def normalize(v):#进行归一化，v是语音特征
    return (v - v.mean(axis=0)) / (v.std(axis=0) + 2e-12)


def audio_feature(x,winlen=0.02,winstep=0.005):

    feature,_ = fbank(x,nfilt=20,winlen=winlen,winstep=winstep,winfunc=np.hamming)
    feature = np.log(feature)
    feature_d1 = delta(feature, N=1)  # 加上两个delta，特征维度X3
    feature_d2 = delta(feature, N=2)

    feature = np.hstack([feature, feature_d1, feature_d2])
    feature = normalize(feature)

    return feature

def get_featrue(f_name,tp='train'):
    x,_ = librosa.load(f_name,sr = 16000)
    sr = 16000
    # print(len(x))
    # joblib.dump(x,'../test_np/%s'%f_name.split("\\")[-1])

    j = -1
    k3 = 0
    while len(x[int(sr * (j+1) * step_len):]) >= sr * win_len:
        j += 1
        k3 += 1
        k1 = x[int(sr * j * step_len):int(sr * (j * step_len + win_len))]
        k = audio_feature(k1,winlen=winlen,winstep=winstep)
        joblib.dump(k,feature_path + '/%s/%d_%s' % (tp, k3,f_name.split("\\")[-1]))
        print(len(k1))
    print(f_name)
    return None



def pool_feature(x):
    path = feature_path +'/%s'%x
    try:
        os.mkdir(path)
    except:
        pass


    if x=='train':
        df = glob('../data/train/*')

    else:
        df = glob('../data/test/*')

    pool=Pool()
    for i in df:
        pool.apply_async(get_featrue,args=(i,x))
    pool.close()
    pool.join()


if __name__ == '__main__':
    pool_feature('train')
    pool_feature('test')
