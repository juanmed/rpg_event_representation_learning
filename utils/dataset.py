import numpy as np
import os
from os import listdir
from os.path import join
import json
import pathlib as pl 
import torch.utils.data

def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events


class NCaltech101:
    def __init__(self, root, augmentation=False):
        self.classes = listdir(root)

        self.files = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)

        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)

        return events, label

class RostrosDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, augmentation = False):
        self.root = root
        self.split = split
        self.augmentation = augmentation

        if (split == "train"):
            metadata_file = "train_metadata.json"
            # read labels
            with open(os.path.join(root,metadata_file), 'r') as json_file:
                self.data = json.load(json_file)
                self.dataset_dir = list(self.data.keys())

        elif(split == "test"):
            metadata_file = "test_metadata.json"
            # read labels
            with open(os.path.join(root,metadata_file), 'r') as json_file:
                self.data = json.load(json_file)
                self.dataset_dir = list(self.data.keys())

        elif(split == "val"):
            raise RunTimeError('Validation dataset still unavailable')
        else:
            raise RuntimeError('{} is not a valid split. Use -test-, -train- or -val-. '.format(self.split))

        # remover archivos 
        self.dataset_dir = [f for f in self.dataset_dir if pl.Path(os.path.join(self.root,f)).is_file()]

    def __len__(self):
        return len(self.dataset_dir)

    def __getitem__(self, idx):

        # get label: Fake : 1, Real: 0
        video_file = self.dataset_dir[idx]
        target = [1. if self.data[video_file]['label'] == 'FAKE' else 0.][0]

        # get event file
        video_dir = os.path.join(self.root, video_file)
        events = np.load(video_dir.replace('.avi','.npy')).astype(np.float32)

        # flip x,y columns... they are in inverse orders
        events[:, 0], events[:, 1] = events[:, 1], events[:, 0].copy()

        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)

        return events, target            

if __name__ == '__main__':
    
    DATASET_DIR = '../../dataset/deepfake-detection-challenge/'
    TRAIN_DIR = 'event_dataset/train'
    TEST_DIR = 'event_dataset/train' # si, tienen el mismo nombre

    train_root = os.path.join(DATASET_DIR, TRAIN_DIR)
    test_root = os.path.join(DATASET_DIR, TEST_DIR)

    train_dataset = RostrosDataset(train_root, split = 'train', augmentation = True)
    test_dataset = RostrosDataset(test_root, split = 'test', augmentation = True)

    n = test_dataset[0]
    print(n)
