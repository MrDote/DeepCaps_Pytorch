'''
Configurations.
'''


import helpers

LEARNING_RATE = 1e-4
COLORS = 1
IMG_SHAPE = 64
NUM_EPOCHS = 200
BATCH_SIZE = 4
LAMBDA_ = 0.2
M_PLUS = 0.9
M_MINUS = 0.1
DECAY_STEP = 20
DECAY_GAMMA = 0.5
CHECKPOINT_NAME = 'deepcaps.pth'
DATASET = 'simard'


LABELS_FILE= f'./Data/votes_deepcaps_{DATASET}.npy'
DATASET_FOLDER = f'./Data/images_deepcaps_{DATASET}_{COLORS}.npy'
CHECKPOINT_FOLDER = './saved_model/'
ACC_FOLDER = './Acc/'


# HEC_FOLDER = './DeepCaps'
# LABELS_FILE= HEC_FOLDER + f'/Data/votes_deepcaps_{DATASET}.npy'
# DATASET_FOLDER = HEC_FOLDER + f'/Data/images_deepcaps_{DATASET}_{COLORS}.npy'
# CHECKPOINT_FOLDER = HEC_FOLDER + '/saved_model/'
# ACC_FOLDER = HEC_FOLDER + '/Acc/'


GRAPHS_FOLDER = None
DEVICE = helpers.get_device()

helpers.check_path(CHECKPOINT_FOLDER)
# helpers.check_path(DATASET_FOLDER)
helpers.check_path(ACC_FOLDER)
# helpers.check_path(GRAPHS_FOLDER)