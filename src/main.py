import configparser
import data
import model
from _processing import *
from sklearn.model_selection import KFold

# config parameters
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')


def run():
    step = int(config.get('BASIC_INFO', 'MODEL_STEP'))
    print('*'*50)
    print('step {}'.format(step))

    '''
        define parameters
    '''
    mal_dir = config.get('PATH', 'MAL_DIR')
    ben_dir = config.get('PATH', 'BEN_DIR')
    label_file = config.get('PATH', 'LABEL_FILE')
    batch_size = int(config.get('CLASSIFIER', 'BATCH_SIZE'))
    k_fold_value = int(config.get('BASIC_INFO', 'K_FOLD_VALUE'))

    '''
        seperate data using K-fold cross validation
    '''
    mal_data = np.array(walk_dir(mal_dir))
    ben_data = np.array(walk_dir(ben_dir))

    cv = KFold(n_splits=k_fold_value, shuffle=True, random_state=0)
    for (train_mal_idx, eval_mal_idx), (train_ben_idx, eval_ben_idx) in zip(cv.split(mal_data), cv.split(ben_data)):
        '''
            load data
        '''
        print('load data')
        train_data = data.DataLoader(mal_data[train_mal_idx], ben_data[train_ben_idx], batch_size=batch_size, mode='train')
        eval_data = data.DataLoader(mal_data[eval_mal_idx], ben_data[eval_ben_idx], batch_size=batch_size, mode='evaluate')

        print('load model')
        model_dic = {
            'epoch': int(config.get('CLASSIFIER', 'EPOCH')),
            'gpu_num': int(config.get('CLASSIFIER', 'GPU_NUM')),
            'keep_prob': float(1 - float(config.get('CLASSIFIER', 'DROPOUT_PROB'))),
            'learning_rate': float(config.get('CLASSIFIER', 'LEARNING_RATE')),
            'model_storage': config.get('CLASSIFIER', 'MODEL_STORAGE'),
            'model_network': config.get('CLASSIFIER', 'NETWORK'),
            'net_input_size': int(config.get('CLASSIFIER', 'INPUT_SIZE')),
            'net_output_size': int(config.get('CLASSIFIER', 'OUTPUT_SIZE')),
            'net_type': config.get('CLASSIFIER', 'NETWORK')
        }

        classifier = model.KISNet(model_num=step,
                                  train_data=train_data,
                                  eval_data=eval_data,
                                  model_dic=model_dic)
        classifier.train()
        classifier.evaluate()
    pass


if __name__ == '__main__':
    run()
    pass
