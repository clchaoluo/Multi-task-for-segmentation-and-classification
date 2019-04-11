import os

class HyperParams:
    """Hyper-Parameters"""
    # data path
    HOME_DIR = os.environ['HOME'] + '/project/residual-attention-network_new_dcmdata/'
    DATASET_DIR = HOME_DIR + "/dataset/"
    SAVED_PATH = HOME_DIR + "trained_models_1/model.ckpt"

    # dataset
    target_dataset = "CIFAR-10"

    # setting
    RANDOM_STATE = 42
    NUM_EPOCHS = 10000
    BATCH_SIZE = 64
    VALID_BATCH_SIZE = 100
    BATCH_SIZE_HEART = 5
    VALID_BATCH_SIZE_HEART = 1
    NUM_EPOCHS_HEART = 100
