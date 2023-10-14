from data_pre import *
from data_pre2 import prepare_data2
import sys
sys.path.append("../")
#from config import *
import numpy as np


np.random.seed(1)
epochs = 1  # training epoch
batch_size = 1000  # training batch size
train_num = 3000  # how many trails for train
test_num = 400  # how many trails for test

# 主要为了增加数据数量，不同的EEG和噪声混合会产生更多的数据，同时也可增强模型鲁棒性，引入不同的噪声仍可还原信号
combin_num = 10  # combin EEG and noise ? times


EEG_all = np.load('/data2/ch/EEGdenoise/data/dataset/dataset2/EEG2s_all_epochs.npy')
EMG_all = np.load('/data2/ch/EEGdenoise/data/dataset/dataset2/EMG2s_all_epochs.npy')
# EEG_all = np.load('/data/ch/EEGdenoise/data/dataset/EEG_all_epochs_512hz.npy')
# EMG_all = np.load('/data/ch/EEGdenoise/data/dataset/EMG_all_epochs_512hz.npy')

EMGEEG_train_input, EMGEEG_train_output, EMGEEG_val_input, EMGEEG_val_output, EMGEEG_test_input, EMGEEG_test_output, test_std_VALUE, SNR_train = prepare_data(EEG_all, EMG_all, combin_num,
                                                                                  0.8, 'EOG')

train_input = EMGEEG_train_input
train_output = EMGEEG_train_output
val_input = EMGEEG_val_input
val_output = EMGEEG_val_output
test_input = EMGEEG_test_input
test_output = EMGEEG_test_output

# np.save('/data/ch/EEGdenoise/data/MA_train_input.npy', train_input)
# np.save('/data/ch/EEGdenoise/data//MA_train_output.npy', train_output)
# np.save('/data/ch/EEGdenoise/data/MA_val_input.npy', val_input)
# np.save('/data/ch/EEGdenoise/data/MA_val_output.npy', val_output)
# np.save('/data/ch/EEGdenoise/data/MA_test_input.npy', test_input)
# np.save('/data/ch/EEGdenoise/data/MA_test_output.npy', test_output)
#np.save('/data/ch/EEGdenoise/data/MA_SNR_train', SNR_train)


np.save('/data2/ch/EEGdenoise/data/dataset/dataset2/EMG_train_input.npy', train_input)
np.save('/data2/ch/EEGdenoise/data/dataset/dataset2/EMG_train_output.npy', train_output)
np.save('/data2/ch/EEGdenoise/data/dataset/dataset2/EMG_val_input.npy', val_input)
np.save('/data2/ch/EEGdenoise/data/dataset/dataset2/EMG_val_output.npy', val_output)
np.save('/data2/ch/EEGdenoise/data/dataset/dataset2/EMG_test_input.npy', test_input)
np.save('/data2/ch/EEGdenoise/data/dataset/dataset2/EMG_test_output.npy', test_output)
