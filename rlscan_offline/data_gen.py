import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt
import copy
import multiprocessing as mp
import time


class DataGen:

    def __init__(
            self,
            datadir="/media/dmitriy/HDD/offline",
            num_workers=10,
            img_dim=(256, 256, 4)
    ):
        self.datadir = datadir
        self._img_keys = {
            'train': [],
            "val": []
        }
        self._labels = {
            'train': {},
            'val': {}
        }
        self.batch_size = {
            'train': 50,
            'val': 50
        }
        self.data_size = {
            'train': 0,
            'val': 0
        }
        self.current_batch_id = {
            'train': 0,
            'val': 0
        }
        self.img_dim = img_dim
        self._batch_buffer_size = 10
        self._terminate = mp.Value('i', 0)
        self._start = {
            'train': mp.Value('i', 0),
            'val': mp.Value('i', 0)
        }
        self.progress_in_epoch = mp.Value('d', 0.0)
        self._build_dataset()
        self._init_shared_variables()
        self._start_workers(num_workers)

    def _init_shared_variables(self):
        """Initializes shared arrays and shared variables."""
        self._new_batch = {'train': {}, 'val': {}}
        self._batch_dict = {'train': {}, 'val': {}}
        self._locks = {'train': {}, 'val': {}}
        self._ix_locks = {'train': mp.Lock(), 'val': mp.Lock()}
        self._permuted_ix = {
            'train': np.frombuffer(mp.Array('i', self.data_size['train']).get_obj(), dtype="int32"),
            'val': np.frombuffer(mp.Array('i', self.data_size['val']).get_obj(), dtype="int32")
        }
        self._ix_processed = {
            'train': np.frombuffer(mp.Array('i', self.data_size['train']).get_obj(), dtype="int32"),
            'val': np.frombuffer(mp.Array('i', self.data_size['val']).get_obj(), dtype="int32")
        }
        for dataset in ["train", "val"]:
            ix = np.random.permutation(self.data_size[dataset]).astype('int32')
            np.copyto(self._permuted_ix[dataset], ix)
            np.copyto(self._ix_processed[dataset], np.ones_like(self._ix_processed[dataset]))
            batch_arr_size = self.batch_size[dataset] * np.product(self.img_dim)
            batch_shape = (self.batch_size[dataset],) + self.img_dim
            for i in range(self._batch_buffer_size):
                self._new_batch[dataset][i] = mp.Value('i', 0)
                data_arr = np.frombuffer(mp.Array('f', int(batch_arr_size)).get_obj(), dtype="float32")
                data_arr = data_arr.reshape(batch_shape)
                labels_arr = np.frombuffer(mp.Array('B', self.batch_size[dataset]).get_obj(), dtype="uint8")
                self._batch_dict[dataset][i] = {'data': data_arr, "labels": labels_arr}
                self._locks[dataset][i] = mp.Lock()

    def _start_workers(self, num_workers):
        """Starts concurrent processes that build data minibatches."""
        self._process_list = []
        for i in range(num_workers):
            p = mp.Process(target=self.prepare_minibatch)
            p.start()
            p_val = mp.Process(target=self.prepare_minibatch, args=('val',))
            p_val.start()
            self._process_list.append(p)
            self._process_list.append(p_val)

    def _build_dataset(self, val_split=0.1):
        with open(os.path.join(self.datadir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        metadata['image_file'] = metadata['image_file']
        self.data_size['train'] = int(len(metadata['image_file']) * (1 - val_split))
        self._img_keys['train'] = metadata['image_file'][:self.data_size['train']]
        self._img_keys['val'] = metadata['image_file'][self.data_size['train']:]
        self.data_size['val'] = len(self._img_keys['val'])
        print("data size", self.data_size)
        for i, img in enumerate(self._img_keys['train']):
            if not metadata['scanner_id'][i] is None:
                self._labels['train'][img] = 1
            else:
                self._labels['train'][img] = 0
        for i, img in enumerate(self._img_keys['val']):
            if not metadata['scanner_id'][self.data_size['train'] + i] is None:
                self._labels['val'][img] = 1
            else:
                self._labels['val'][img] = 0
        print("class 1 train", np.sum(list(self._labels['train'].values())), "class 0 train", len(self._labels['train']) - np.sum(list(self._labels['train'].values())))
        print("class 1 val", np.sum(list(self._labels['val'].values())), "class 0 val", len(self._labels['val']) - np.sum(list(self._labels['val'].values())))
        return metadata

    def prepare_minibatch(self, dataset="train"):
        """Builds minibatches and stores them to shared memory.

        This function is run by concurrent processes.
        """

        if dataset == "val":
            cvb = 1
        while not self._terminate.value:
            for id in self._new_batch[dataset]:
                if not self._new_batch[dataset][id].value:
                    locked = self._locks[dataset][id].acquire(False)
                    if not locked:
                        continue
                    locked = self._ix_locks[dataset].acquire(timeout=1.0)
                    if not locked:
                        print("failed to get a lock")
                        continue
                    if self.data_size[dataset] - self._start[dataset].value < self.batch_size[dataset]:
                        print("Reached the end of the dataset {}, resetting".format(dataset))
                        ix = np.random.permutation(self.data_size[dataset]).astype('int32')
                        np.copyto(self._permuted_ix[dataset], ix)
                        np.copyto(self._ix_processed[dataset], np.zeros(self._ix_processed[dataset].shape, dtype="int32"))
                        self._start[dataset].value = 0
                    #np.copyto(self._ix_processed[dataset][self._start[dataset]: self._start[dataset] + self.batch_size[dataset]], np.ones(self.batch_size[dataset], dtype="int32"))
                    ix_range = self._permuted_ix[dataset][self._start[dataset].value: self._start[dataset].value + self.batch_size[dataset]]
                    self._start[dataset].value += self.batch_size[dataset]
                    self._ix_locks[dataset].release()
                    data, labels = self.load_batch(ix_range, dataset=dataset)
                    data = np.array(data)
                    #data = np.expand_dims(np.array(data), axis=-1)
                    labels = np.array(labels, dtype='uint8')
                    np.copyto(self._batch_dict[dataset][id]["data"], data)
                    np.copyto(self._batch_dict[dataset][id]["labels"], labels)
                    self._new_batch[dataset][id].value = 1
                    #print("stored batch into id", id)
                    batches_aval = sum([self._new_batch[dataset][i].value for i in self._new_batch[dataset]])
                    #print("start", dataset, self._start[dataset].value)
                    if dataset == "train" and batches_aval < 8:
                        print("Prepared new batch, {} batches available".format(batches_aval))
                    self.progress_in_epoch.value = (self._start[dataset].value + 0.0)/len(self._permuted_ix[dataset])
                    #break
                    self._locks[dataset][id].release()
            #print("Im sleeping")
            time.sleep(0.1)

    def generate_batch(self, dataset="train"):
        """A generator reading and returning data batches from shared memory.

        To be used by an external training function.
        """
        id = self.current_batch_id[dataset]
        while True:
            if self._new_batch[dataset][id].value:
                #print("reading batch from id", id)
                ix = np.random.permutation(self._batch_dict[dataset][id]["data"].shape[0])
                data = np.copy(self._batch_dict[dataset][id]["data"])[ix]
                labels = np.copy(self._batch_dict[dataset][id]["labels"])[ix].astype("float32")
                self._new_batch[dataset][id].value = 0
                self.current_batch_id[dataset] += 1
                if not self.current_batch_id[dataset] in self._new_batch[dataset]:
                    self.current_batch_id[dataset] = 0
                yield data, labels
            else:
                time.sleep(0.01)

    def load_batch(self, ix, dataset='train'):
        """Loads a batch of images defined by indices in ix list."""
        images = []
        labels = []
        for i in ix:
            filename = self._img_keys[dataset][i]
            img = self.load_image(filename)
            images.append(img)
            img[0, 0, 0] = i + 0.0
            labels.append(self._labels[dataset][filename])
            # if labels[-1] == 0:
            #     plt.figure(figsize=(10, 2.8))
            #     for i in range(4):
            #         plt.subplot(1, 4, i + 1)
            #         plt.imshow(img[:, :, i] - np.mean(img[:, :, i]), cmap='gray')
            #         plt.xticks([])
            #         plt.yticks([])
            #     plt.subplots_adjust(left=0, right=1)
            #     plt.show()
        return images, labels

    def load_image(self, filename):
        filename = filename[filename.find("offline"):]
        filename = filename[filename.find("/"):][1:]
        with open(os.path.join(self.datadir, filename), "rb") as f:
            jpg_frames = pickle.load(f)
        imgs = [np.array(Image.open(jpg)) for jpg in jpg_frames]
        imgs = np.stack(imgs).astype('float32')/255.0
        imgs = np.transpose(imgs, axes=[1, 2, 0])
        return imgs


if __name__ == "__main__":
    d = DataGen()
    for mb in d.generate_batch("train"):
        data, labels = mb
        plt.figure(figsize=(20, 12))
        count = 0
        for i in range(len(data)):
            if labels[i] == 0:
                continue
            for j in range(4):
                plt.subplot(4, 4, 4 * count + j + 1)
                plt.imshow(data[i, j, :, :, 0], cmap='gray')
            count += 1
            if count == 4:
                break
        plt.show()
        cvb = 1
