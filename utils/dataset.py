import numpy as np
import os
from os import listdir
from os.path import join
import json
import pathlib as pl 
import torch.utils.data
import random

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
        self.dataset_dir= []

        if (split == "train"):
            metadata_file = "train_sample.json"

            # read labels
            with open(os.path.join(root,metadata_file), 'r') as json_file:
                self.data = json.load(json_file)
                self.dataset_dir = list(self.data.keys())

        elif(split == "test"):
            metadata_file = "test_sample.json"
            # read labels
            with open(os.path.join(root,metadata_file), 'r') as json_file:
                self.data = json.load(json_file)
                self.dataset_dir = list(self.data.keys())

        elif(split == "val"):
            self.dataset_dir = os.listdir(self.root)
        else:
            raise RuntimeError('{} is not a valid split. Use -test-, -train- or -val-. '.format(self.split))

        # remover archivos listados en json pero que no existen
        self.dataset_dir = [f for f in self.dataset_dir if pl.Path(os.path.join(self.root,f.replace(".avi", ".npy"))).is_file()]
        self.dataset_dir = [f for f in self.dataset_dir if f.endswith(".npy")]
        self.index = 0
    def __len__(self):
        return len(self.dataset_dir)

    def __getitem__(self, idx):

        # get label: Fake : 1, Real: 0
        success = False
        while not success:
            file = self.dataset_dir[self.index]
            try:
                file_dir = os.path.join(self.root, file)
                events = np.load(file_dir, allow_pickle = True).astype(np.float32)
                events[-1,-1]
                if events.shape[0] == 0:
                    raise Exception("Archivo sin eventos.")
                else:
                    success = True
                    break
                    break
            except Exception as e:
                print("ERROR: {}".format(str(e)))
                try:
                    os.remove(file)
                except Exception as e:
                    print("ERRO AL ELIMINAR: {}".format(str(e)))
                self.index = self.index + 1
                if self.index >= len(self.dataset_dir):
                    self.index = 0

        if self.split == "val":
            target = [1. if (random.random() > 0.5) else 0.][0]
        else:
            target = [1. if self.data[file]['label'] == 'FAKE' else 0.][0]

        # get event file
        #file_dir = os.path.join(self.root, file)
        #events = np.load(file_dir, allow_pickle = True).astype(np.float32)
        #if events.shape[0] > 2000000:
        #    events = events[:2000000]

        # flip x,y columns... they are in inverse orders
        # not anymore, solvd 
        #events[:, 0], events[:, 1] = events[:, 1], events[:, 0].copy()

        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)

        self.index = self.index + 1
        if self.index >= len(self.dataset_dir):
            self.index = 0

        return events, int(target)            


class RostrosDatasetEnorme(torch.utils.data.Dataset):

    def __init__(self, root, split, augmentation = False):
        self.root = root
        self.split = split
        self.augmentation = augmentation
        self.files= []
        self.labels = []

        if (split == "train"):
            metadata_file = "train_metadata.json"

            folders = os.listdir(self.root)
            folders = [f for f in folders if os.path.isdir(os.path.join(self.root,f)) and (not f.endswith(".csv"))]# and ( f != 'dfdc_train_part_39') ]

            for folder in folders:
                # read labels
                with open(os.path.join(self.root,folder,metadata_file), 'r') as json_file:
                    #print(os.path.join(self.root,folder,metadata_file))
                    data = json.load(json_file)
                
                file_names = list(data.keys())
                self.files  +=  [os.path.join(self.root,folder,file) for file in  file_names]
                self.labels +=  [data[file]['label'] for file in file_names]

        elif(split == "test"):
            metadata_file = "test_metadata.json"

            folders = os.listdir(self.root)
            folders = [f for f in folders if os.path.isdir(os.path.join(self.root,f)) and (not f.endswith(".csv"))]# and ( f != 'dfdc_train_part_39')]

            for folder in folders:
                # read labels
                with open(os.path.join(self.root,folder,metadata_file), 'r') as json_file:
                    #print(os.path.join(self.root,folder,metadata_file))
                    data = json.load(json_file)
                
                file_names = list(data.keys())
                self.files  +=  [os.path.join(self.root,folder,file) for file in  file_names]
                self.labels +=  [data[file]['label'] for file in file_names]

        elif(split == "val"):
            self.files = os.listdir(self.root)
        else:
            raise RuntimeError('{} is not a valid split. Use -test-, -train- or -val-. '.format(self.split))

        of = len(self.files)
        ol = len(self.labels)

        # remover archivos listados en json pero que no existen
        real_files = []
        real_labels = []
        for i,file in enumerate(self.files):
            if pl.Path(file).is_file():
                real_files.append(file)
                real_labels.append(self.labels[i])
            else:
                print("{} does not exist. ".format(file))

        self.files=real_files
        self.labels=real_labels

        ff = len(self.files)
        fl = len(self.labels)
        assert ff == fl, "Luego de eliminar archivos, hay {} archivos pero {} etiquetas".format(ff,fl)

        # juntos y revueltos
        c = list(zip(self.files, self.labels))
        random.shuffle(c)
        self.files, self.labels = zip(*c)

        shouldnot = ['oidthpmdmt4.npy','fglfkwlujv2.npy','atbxybzopa6.npy','kacpwvfkmq6.npy','cssuukpkgx0.npy','wxzeidnrpx2.npy','zjhmmibena2.npy']
        for i in shouldnot:
            if i in self.files: 
                print(" +++++++++++++ {} ESTA +++++++++++++++".format(i))

        print("En {} Habia {} files, {} labels. Luego de filtro: {} files, {} labels".format(self.split,of,ol,ff,fl))

        self.index = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        success = False
        while not success:
            file = self.files[self.index]
            try:
                events = np.load(file, allow_pickle = True).astype(np.float32)
                if events.shape[0] == 0:
                    raise Exception("Archivo sin eventos.")
                else:
                    success = True
                    break
                    break
            except Exception as e:
                print("ERROR: {}".format(str(e)))
                try:
                    os.remove(file)
                except Exception as e:
                    print("ERRO AL ELIMINAR: {}".format(str(e)))
                self.index = self.index + 1
                if self.index >= len(self.files):
                    self.index = 0

        if self.split == "val":
            target = [1. if (random.random() > 0.5) else 0.][0]
            print(file)
        else:
            # get label: Fake : 1, Real: 0
            target = target = [1. if self.labels[self.index] == 'FAKE' else 0.][0]

        # get event file
        #file_dir = os.path.join(self.root, file)
        

        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)

        self.index = self.index + 1
        if self.index >= len(self.files):
            self.index = 0
        return events, int(target)            

if __name__ == '__main__':
    
    DATASET_DIR = '/media/jfmy/8TB HDD/JUAN/dfdc_event_all'
    TRAIN_DIR = 'dfdc_train_part_11'
    TEST_DIR = 'dfdc_train_part_11' # si, tienen el mismo nombre... estan en el mismo folder

    train_root = os.path.join(DATASET_DIR, TRAIN_DIR)
    test_root = os.path.join(DATASET_DIR, TEST_DIR)

    train_dataset = RostrosDatasetEnorme(DATASET_DIR, split = 'train', augmentation = True)
    test_dataset = RostrosDatasetEnorme(DATASET_DIR, split = 'test', augmentation = False)

    print("Train dataset: {}".format(len(train_dataset)))
    print("Test dataset: {}".format(len(test_dataset)))

    for i, events in enumerate(train_dataset):
        print(len(events[0]), events[1])
    #print("Testing")
    #for i, events in enumerate(test_dataset):
    #    print(len(events[0]), events[1])
