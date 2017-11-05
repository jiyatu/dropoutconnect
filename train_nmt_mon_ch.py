import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    basedir = './data'
    validerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        n_words=7200,
                                        n_words_src=82000,
                                        decay_c=params['decay-c'][0],
                                        clip_c=params['clip-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0],
                                        maxlen=50,
                                        batch_size=32,
                                        valid_batch_size=64,
					datasets=['%s/train.clean.mo'%basedir,
					'%s/train.clean.ch'%basedir],
					valid_datasets=['%s/dev.clean.mo'%basedir,
					'%s/dev.clean.ch'%basedir],
					dictionaries=['%s/train.clean.mo.pkl'%basedir,
					'%s/train.clean.ch.pkl'%basedir],
                                        validFreq=1500,
                                        dispFreq=500,#display ecpoch 0 Cost every dispFreq times
                                        saveFreq=1500,
                                        sampleFreq=1500,
                                        use_dropout=params['use-dropout'][0],
                                        overwrite=False)
    return validerr

if __name__ == '__main__':
    basedir = './'
    main(0, {
        'model': ['%s/models/model_session3.npz'%basedir],
        'dim_word': [1000],
        'dim': [1000],
        'n-words': [20000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.001],
        'reload': [True]})


