import numpy as np
import keras
from keras.layers import *
import tensorflow as tf
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.models import Model
import time
import datetime
from keras.applications import resnet50, densenet, nasnet, mobilenet_v2
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from keras.regularizers import l2
import argparse


from rlscan_offline.data_gen import DataGen
from keras.callbacks import LambdaCallback, ModelCheckpoint


class KerasDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_gen, n_channels=1,
                 n_classes=2, shuffle=True, dataset="train"):
        'Initialization'
        self.data_gen = data_gen
        self.dim = data_gen.img_dim
        self.batch_size = data_gen.batch_size[dataset]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.dataset = dataset
        self.epoch_proc_time = time.time()
        self.ix_count = np.zeros(self.data_gen.data_size[dataset])

    def __len__(self):
        'Denotes the number of batches per epoch'
        batches_per_epoch = self.data_gen.data_size[self.dataset]//self.data_gen.batch_size[self.dataset]
        if self.dataset == "train":
            if self.data_gen.data_size[self.dataset] > 1e6:
                batches_per_epoch //= 20
            else:
                batches_per_epoch //= 3
        return batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        print("epoch processing time", time.time() - self.epoch_proc_time)
        start_time = time.time()
        data, labels = next(self.data_gen.generate_batch(dataset=self.dataset))
        print("epoch acquisition time", time.time() - start_time)
        for i, im in enumerate(data):
            self.ix_count[int(im[0, 0, 0])] += 1
            data[i, 0, 0, 0] = 0
        labels = to_categorical(labels, num_classes=self.n_classes)
        #labels[labels == 0] = 0.01
        #labels[labels == 1] = 0.99
        if self.dataset == "val":
            print("val counts", np.sum(self.ix_count == 1))
        elif self.dataset == "train":
            print("train counts", np.sum(self.ix_count == 1))
        #print("batch index", index)
        self.epoch_proc_time = time.time()
        return data, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        cvn = 2
        print("train counts", np.sum(self.ix_count == 1))
        self.ix_count.fill(0.0)
        return


class LabelDistribution(keras.metrics.Metric):
    """
    Computes the per-batch label distribution (y_true) and stores the array as
    a metric which can be accessed via keras CallBack's

    :param n_class: int - number of distinct output class(es)
    """

    def __init__(self, **kwargs):
        super(LabelDistribution, self).__init__(**kwargs)


    def update_state(self, y_true, y_pred, sample_weight=None):
        self.y_true = y_true
        self.y_pred = y_pred

    def result(self):
        return self.y_true, self.y_pred

    def reset_states(self):
        return


class ResnetModel:

    def __init__(self,
                 num_classes=2,
                 learning_rate=1e-4,
                 epochs=15,
                 aggregate_grads=True,
                 gpu=0,
                 separate_validation=False,
                 datadir=""
                 ):
        self.data_gen = DataGen(separate_validation=separate_validation, datadir=datadir)
        self.keras_data_gen_train = KerasDataGenerator(data_gen=self.data_gen, dataset="train")
        self.keras_data_gen_val = KerasDataGenerator(data_gen=self.data_gen, dataset="val")
        self.input_shape = self.data_gen.img_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cumulative_grads = []
        self.grads_collected = 0
        self.aggregate_grads = aggregate_grads
        mydevice = "/gpu:{}".format(gpu)
        with tf.device(mydevice):
        #with strategy.scope():
            self.init_session()
            self.build_model()


    def build_model(self):
        """Builds the network symbolic graph in tensorflow."""
        self.img = Input(name="input", shape=self.input_shape, dtype='float32')
        x = self.img
        x = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4),
                   activation="relu",
                   padding='same'))(self.img)
        x = BatchNormalization()(x)
        #x = TimeDistributed(Dropout(0.5))(x)
        x = TimeDistributed(Conv2D(64, (5, 5), strides=(2, 2),
                   activation="relu",
                   padding='same'))(x)
        x = BatchNormalization()(x)
        #x = TimeDistributed(Dropout(0.5))(x)
        # x = TimeDistributed(Conv2D(128, (5, 5), strides=(2, 2),
        #            activation="relu",
        #            padding='same'))(x)
        # # x = TimeDistributed(Dropout(0.5))(x)
        # x = TimeDistributed(Conv2D(128, (5, 5), strides=(2, 2),
        #            activation="relu",
        #            padding='same'))(x)
        # # x = TimeDistributed(Dropout(0.5))(x)
        # x = TimeDistributed(Conv2D(128, (5, 5), strides=(2, 2),
        #            activation="relu",
        #            padding='same'))(x)
        outs = Lambda(lambda x: tf.unstack(x, axis=1))(x)
        new_outs = []
        for i, x in enumerate(outs):
            # x = Conv2D(32, (8, 8), strides=(4, 4),
            #                                  activation="relu",
            #                                  padding='same')(x)
            # x = Conv2D(64, (5, 5), strides=(2, 2),
            #                                  activation="relu",
            #                                  padding='same')(x)
            x = Conv2D(128, (5, 5), strides=(2, 2),
                                             activation="relu",
                                             padding='same')(x)
            x = BatchNormalization()(x)
            #x = TimeDistributed(Dropout(0.5))(x)
            x = Conv2D(128, (5, 5), strides=(2, 2),
                                             activation="relu",
                                             padding='same')(x)
            x = BatchNormalization()(x)
            #x = TimeDistributed(Dropout(0.5))(x)
            x = Conv2D(128, (5, 5), strides=(2, 2),
                                             activation="relu",
                                             padding='same')(x)
            x = BatchNormalization()(x)
            #x = TimeDistributed(Dropout(0.5))(x)
            #x = Flatten()(x)
            x = GlobalMaxPooling2D()(x)
            new_outs.append(x)
        # x = densenet.DenseNet121(include_top=False,
        #                         weights=None,
        #                         #input_tensor=x,
        #                         input_shape=(self.input_shape),
        #                         pooling="max")(self.img)
        #x = TimeDistributed(Flatten())(x)
        #x = Lambda(lambda x: tf.reshape(x, [-1, x.shape[1] * x.shape[2]]))(x)
        x = Concatenate(axis=-1)(new_outs)
        # x = Dense(128, activation='relu', name='lin1')(x)
        #x = Dropout(0.5)(x)
        self.output = Dense(self.num_classes, activation="softmax")(x)
        self.model = Model(inputs=self.img, outputs=self.output)
        custom_metrics = LabelDistribution()
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

    def init_session(self):
        config = tf.ConfigProto(
            allow_soft_placement=True
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.__enter__()

    def train(self):
        self.all_losses = {'train': [], 'val': []}
        self.accs = {'val_sens': [], 'val_spec': [], 'val_sens_single': [], 'val_spec_single': []}
        # val_sens_single, val_spec_single = self.evaluate_on_validation_singles_set()
        # print("Validation singles accuracy: sensitivity {}, specificity {}".format(val_sens_single, val_spec_single))
        # self.evaluate_on_test_singles_set()
        self.max_acc = 0.71
        def print_logs(epoch, logs):
            #val_loss = self.evaluate_on_validation_set()
            #val_sens_single, val_spec_single = self.evaluate_on_validation_singles_set()
            #print("Loss: train {}, validation {}".format(logs['loss'], logs['val_loss']))
            # print("Validation accuracy: sensitivity {}, specificity {}".format(val_sens, val_spec))
            #print("Validation singles accuracy: sensitivity {}, specificity {}".format(val_sens_single, val_spec_single))
            print("val count", np.sum(self.keras_data_gen_val.ix_count == 1))
            self.keras_data_gen_val.ix_count.fill(0.0)
            self.all_losses['train'].append(logs['loss'])
            self.all_losses['val'].append(logs['val_loss'])
            if logs['val_accuracy'] >= self.max_acc:
                self.max_acc = logs['val_accuracy']
                print("got 0.71 validation")
                start = time.time()
                self.evaluate_on_validation_set()
                print("eval time", time.time() - start)
                self.save()
            #self.save()
        on_epoch_end = LambdaCallback(on_epoch_end=lambda epoch, logs: print_logs(epoch, logs))
        save_callback = ModelCheckpoint(filepath="saved_models/padded_nasnet_keras_model_{}.pkl".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),
                                        save_weights_only=True,
                                        period=1
                                        )
        self.model.fit(x=self.keras_data_gen_train,
                       epochs=self.epochs,
                       validation_data=self.keras_data_gen_val,
                       #steps_per_epoch=100 // self.data_gen.batch_size['train'],
                       #validation_steps=val_steps,
                       workers=0,
                       verbose=2,
                       callbacks=[
                           on_epoch_end,
                           #save_callback
                       ]
                       )

    def evaluate_on_validation_set(self):
        """Evaluate on orders combined from images from validation set."""
        #predictions = []
        class_labels = []
        predictions = []
        all_logits = []
        batches_processed = 0
        for mb in self.data_gen.generate_batch(dataset="val"):
            data, labels = mb
            logits = self.model.predict(data)
            #loss, acc = self.model.evaluate(data, to_categorical(labels, num_classes=self.num_classes), verbose=0)
            predictions.append(np.argmax(logits, axis=-1) == labels)
            class_labels.append(labels)
            all_logits.append(logits[:, 1])
            #preds.append(preds)
            batches_processed += 1
            if batches_processed > self.data_gen.data_size['val']//self.data_gen.batch_size['val']:
                break
        predictions = np.hstack(predictions)
        class_labels = np.hstack(class_labels)
        all_logits = np.hstack(all_logits)
        sens = np.sum(predictions[class_labels == 1]) / np.sum(class_labels == 1)
        spec = np.sum(predictions[class_labels == 0]) / np.sum(class_labels == 0)
        acc = np.mean(predictions)
        fpr, tpr, thresholds = roc_curve(class_labels, all_logits)
        auc = roc_auc_score(class_labels, all_logits)
        print("Validation acc, sens, spec, roc", acc, sens, spec, auc)
        cvb = 1
        #return np.mean(losses)

    def build_roc(self, labels, scores):
        scores_sorted = sorted(scores)
        sens = []
        spec = []
        for i in range(len(scores_sorted)):
            pos_pred = scores >= scores_sorted[i]
            neg_pred = scores < scores_sorted[i]
            sens.append(np.sum((labels==1) * (pos_pred))/np.sum(labels==1))
            spec.append(1 - np.sum((labels==0) * neg_pred)/np.sum(labels==0))
        auc = np.mean(sens)
        return spec, sens, auc

    def save(self):
        import pickle
        var_dict = {}
        for var in tf.global_variables():
            if 'model' in var.name:
                var_dict[var.name] = var.eval()
        with open("saved_models/cnn_shared_5_layers_{}_{}.pkl".format(self.max_acc, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), "wb") as f:
            pickle.dump({'weights': self.model.get_weights(),
                         "losses": self.all_losses,
                         "val_acc": self.accs}, f)

    def load(self, filename):
        import pickle
        with open(filename, "rb") as f:
            model_dict = pickle.load(f)
        # assign_ops = []
        # for var in tf.global_variables():
        #     if 'model' in var.name:
        #         assign_ops.append(var.assign(model_dict['weights'][var.name]))
        # sess = tf.get_default_session()
        # sess.run(assign_ops)
        self.model.set_weights(model_dict['weights'])
        return model_dict

    def show_pictures(self):
        import matplotlib.pyplot as plt
        imgs_to_show = []
        labels_to_show = []
        logits_to_show = []
        im_num = 1
        for mb in self.data_gen.generate_batch(dataset="val"):
            data, labels = mb
            logits = self.model.predict(data)
            # loss, acc = self.model.evaluate(data, to_categorical(labels, num_classes=self.num_classes), verbose=0)

            for i in range(len(data)):
                data[i, 0, 0, 0] = 0
                if logits[i, 1] < 0.1 or logits[i, 1] > 1.8:
                    imgs_to_show.append(data[i])
                    labels_to_show.append(labels[i])
                    logits_to_show.append(logits[i])
                    if len(imgs_to_show) >= im_num:
                        break
            if len(imgs_to_show) >= im_num:
                plt.figure(figsize=(20, 12))
                for i in range(im_num):
                    np.save("images/current_im.npy", imgs_to_show[i])
                    for j in range(4):
                        plt.subplot(1, 4, i * 4 + j + 1)
                        plt.imshow(imgs_to_show[i][j, :, :, 0], cmap='gray')
                        plt.xticks([])
                        plt.yticks([])
                        plt.title("{}, {:.3f}".format(labels_to_show[i], logits_to_show[i][1]))
                plt.show()

            imgs_to_show = []
            labels_to_show = []
            logits_to_show = []

    def show_score_hist(self):
        all_labels = []
        all_logits = []
        batches_processed = 0
        for mb in self.data_gen.generate_batch(dataset="val"):
            data, labels = mb
            logits = self.model.predict(data)
            all_labels.append(labels)
            all_logits.append(logits[:, 1])
            batches_processed += 1
            if batches_processed > self.data_gen.data_size['val'] // self.data_gen.batch_size['val']:
                break
        all_labels = np.hstack(all_labels)
        all_logits = np.hstack(all_logits)
        fpr, tpr, thresholds = roc_curve(all_labels, all_logits)
        auc = roc_auc_score(all_labels, all_logits)
        plt.figure()
        plt.hist(all_logits[all_labels== 1], bins=20, alpha=1, label="scan success")
        plt.hist(all_logits[all_labels== 0], bins=20, alpha=0.7, label="scan fail")
        plt.legend()
        plt.grid()
        plt.title("Predicted score distribution")

        plt.figure()
        plt.plot(fpr, tpr,
                 label="AUC: {:.3f}".format(auc))
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RLScan offline analysis')
    parser.add_argument('--model', type=str,
                        default='resnet')  # 'lstm', 'padded_lstm', 'scapegoat', 'weighted_scapegoat' or 'comb_repr'
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    model = ResnetModel(epochs=args.epochs, datadir=args.datadir, gpu=args.gpu)
    #model.load("saved_models/cnn_shared_5_layers_0.714248776435852_2021-02-01-10-00-25.pkl")
    #model.evaluate_on_validation_set()
    #model.show_pictures()
    #model.show_score_hist()
    model.train()
    #model.save()

