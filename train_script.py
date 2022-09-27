import os
# Standard
import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse
import wandb
from types import SimpleNamespace

# Utils
import h5py

# Deep Learning
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

# User defined
from architectures.laddernet import LadderNet
from architectures.unet import UNet
from metrics.multiclass_dice import multiclass_dice

config_defaults = SimpleNamespace(
    model='Unet',
    subset='all',
    batch_size=5,
    epochs=1,
    lr=1e-3,
    dataset='data/image_dataset.hdf5',
    wandb_project='ai4mi',
)

# f = h5py.File("../data/image_dataset.hdf5", "r")


# frames2ch = np.append(f["train 2ch frames"][:,:,:,:], f["train 4ch frames"][:,:,:,:], axis=0)
# masks2ch = np.append(f["train 2ch masks"][:,:,:,:], f["train 4ch masks"][:,:,:,:], axis=0)

# train_frames, test_frames, train_masks, test_masks = train_test_split(frames2ch, masks2ch)


def parse_args():
    parser = argparse.ArgumentParser(description='Config for training')
    parser.add_argument('--model', type=str, default=config_defaults.model, choices=['Unet', 'Laddernet'])
    parser.add_argument('--dataset', type=str, default=config_defaults.dataset)
    parser.add_argument('--subset', type=str, default=config_defaults.subset, choices=['4ch', '2ch', 'all'])
    parser.add_argument('--epochs', type=int, default=config_defaults.epochs)
    parser.add_argument('--lr', type=float, default=config_defaults.lr)
    parser.add_argument('--batch_size', type=int, default=config_defaults.batch_size)


    return parser.parse_args()

def get_dataset(config):
    f =  h5py.File(config.dataset, "r")
    match config.subset:
        case '4ch':
            train_frames = f["train 4ch frames"]
            train_masks = f["train 4ch masks"]
            test_frames = f["test 4ch frames"]
            test_masks = f["test 4ch masks"]
        case '2ch':
            train_frames = f["train 2ch frames"]
            train_masks = f["train 2ch masks"]
            test_frames = f["test 4ch frames"]
            test_masks = f["test 4ch masks"]
        case 'all':
            train_frames = np.append(f["train 2ch frames"][:,:,:,:], f["train 4ch frames"][:,:,:,:], axis=0)
            train_masks = np.append(f["train 2ch masks"][:,:,:,:], f["train 4ch masks"][:,:,:,:], axis=0)
            test_frames = np.append(f["test 2ch frames"][:,:,:,:], f["test 4ch frames"][:,:,:,:], axis=0)
            test_masks = np.append(f["test 2ch masks"][:,:,:,:], f["test 4ch masks"][:,:,:,:], axis=0)
        
    return train_frames, train_masks, test_frames, test_masks

def get_model(config):
    match config.model:
        case 'Unet':
            model = UNet(input_size=(384, 384, 1), depth=5, num_classes=4, filters=10, batch_norm=True)
        case 'Laddernet':
            model = LadderNet(input_size=(384, 384, 1), num_classes=4, filters=20)
    
    model.compile(optimizer=Adam(lr=config.lr), loss="sparse_categorical_crossentropy", metrics=[multiclass_dice, "accuracy"])
    return model

def train(model, data, config):
    earlystop = EarlyStopping(monitor='val_multiclass_dice', min_delta=0, patience=5,
                          verbose=1, mode="max", restore_best_weights = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_multiclass_dice', factor=0.2, patience=2,
                              verbose=1, mode="max", min_lr=1e-5)

    train_frames, train_masks, test_frames, test_masks = data

    history = model.fit(x=train_frames,
                y=train_masks,
                validation_data=[test_frames, test_masks],
                batch_size=config.batch_size,
                epochs=config.epochs,
                callbacks=[earlystop, reduce_lr])

    return history

def save_results(config, history):
    results_dict = {'config': config, 'logs': history.history}

def main(config=config_defaults):
    data = get_dataset(config)
    model = get_model(config)
    history = train(model, data, config)
    save_results(config, history, model)

    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
