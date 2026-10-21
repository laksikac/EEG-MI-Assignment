"""
Experiment config for HWNet (BCIC2a, subject-dependent), band-power features.

Part 1 saves features of shape (n_trials, 20, 4). With data_format=None the
DataLoader returns them as (n, C, T) = (n, 20, 4), which is HWNet's input_shape.
"""

from mixnet.models import *
from mixnet.loss import *
from mixnet.utils import dotdict
from tensorflow.keras.optimizers import Adam


def get_params(dataset, train_type, data_type, num_class, num_chs=None,
               loss_weights=None, log_dir='logs', **kwargs):

    model_name = 'canHWNet'
    n_subjects = 9 if dataset == 'BCIC2a' else \
                 9 if dataset == 'BCIC2b' else 0

    n_chs = 20 if num_chs is None else num_chs
    n_bands = 4
    input_shape = (12,19,17)

    loss_weights = [1.] if loss_weights is None else loss_weights

    log_path = '{}/{}/{}_{}_classes_{}'.format(log_dir, model_name, train_type,
                                               str(num_class), dataset)

    # small net + tiny dataset -> modest lr / batch size, strong dropout
    factor = 0.5
    es_patience = 20
    lr = 1e-3
    min_lr = 1e-5
    batch_size = 16
    patience = 5
    epochs = 200
    min_epochs = 0
    dropout_rate = 0.5

    params = dotdict({
        'model': HWNet,
        'model_params': dotdict({
            'model_name': model_name,
            'input_shape': (28,4,71),
            'hidden': (32, 16),             # units of each hidden Dense layer
            'class_balancing': True,
            'f1_average': 'macro',
            'num_class': num_class,
            'loss': [SparseCategoricalCrossentropy()],
            'loss_names': ['crossentropy'],
            'loss_weights': loss_weights,
            'epochs': epochs,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'optimizer': Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
            'lr': lr,
            'min_lr': min_lr,
            'factor': factor,
            'patience': patience,
            'es_patience': es_patience,
            'min_epochs': min_epochs,
            'verbose': 1,
            'log_path': log_path,
            'data_format': 'channels_first',
        }),

        'data_params': dotdict({
            'dataset': dataset,
            'train_type': train_type,
            'data_format': None,            # (n, 20, 4) stays as-is
            'data_type': data_type,
            'num_class': num_class,
            'dataset_path':'/workspace/can_eeg/SFFT/experiments/datasets',
            'n_subjects': n_subjects,
            'n_folds': 5,
            'load_path': '/workspace/can_eeg/SFFT/experiments/datasets/BCIC2a/HW_prep/2_class/subject_dependent',
        }),

        'log_path': log_path,
    })

    return params