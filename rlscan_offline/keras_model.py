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
strategy = tf.distribute.MirroredStrategy()


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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.data_gen.data_size[self.dataset]//self.data_gen.batch_size[self.dataset]

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        data, labels = next(self.data_gen.generate_batch(dataset=self.dataset))
        labels = to_categorical(labels, num_classes=self.n_classes)
        return data, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
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
                 learning_rate=3e-4,
                 epochs=15,
                 opt_batch=40,
                 aggregate_grads=True,
                 gpu=0
                 ):
        self.data_gen = DataGen()
        self.keras_data_gen_train = KerasDataGenerator(data_gen=self.data_gen, dataset="train")
        self.keras_data_gen_val = KerasDataGenerator(data_gen=self.data_gen, dataset="val")
        self.input_shape = self.data_gen.img_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.opt_batch = opt_batch
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
        x = TimeDistributed(Conv2D(32, (5, 5), strides=(2, 2),
                                         activation="relu",
                                         padding='same'))(self.img)
        x = TimeDistributed(Conv2D(64, (5, 5), strides=(2, 2),
                                         activation="relu",
                                         padding='same'))(x)
        x = TimeDistributed(Conv2D(64, (5, 5), strides=(2, 2),
                                         activation="relu",
                                         padding='same'))(x)
        x = TimeDistributed(Conv2D(64, (5, 5), strides=(2, 2),
                                         activation="relu",
                                         padding='same'))(x)
        x = TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2),
                                         activation="relu",
                                         padding='same'))(x)
        x = TimeDistributed(Flatten())(x)
        x = Lambda(lambda x: tf.reshape(x, [-1, x.shape[1] * x.shape[2]]))(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', name='lin1')(x)
        self.output = Dense(self.num_classes, activation="softmax")(x)
        self.model = Model(inputs=self.img, outputs=self.output)
        custom_metrics = LabelDistribution()
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy", custom_metrics],
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
        def print_logs(epoch, logs):
            #val_loss = self.evaluate_on_validation_set()
            #val_sens_single, val_spec_single = self.evaluate_on_validation_singles_set()
            #print("Loss: train {}, validation {}".format(logs['loss'], logs['val_loss']))
            # print("Validation accuracy: sensitivity {}, specificity {}".format(val_sens, val_spec))
            #print("Validation singles accuracy: sensitivity {}, specificity {}".format(val_sens_single, val_spec_single))
            self.all_losses['train'].append(logs['loss'])
            self.all_losses['val'].append(logs['val_loss'])
            #self.save()
        on_epoch_end = LambdaCallback(on_epoch_end=lambda epoch, logs: print_logs(epoch, logs))
        save_callback = ModelCheckpoint(filepath="saved_models/padded_nasnet_keras_model_{}.pkl".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),
                                        save_weights_only=True,
                                        period=1
                                        )
        self.model.fit(x=self.keras_data_gen_train,
                       epochs=100,
                       validation_data=self.keras_data_gen_val,
                       #steps_per_epoch=100 // self.data_gen.batch_size['train'],
                       #validation_steps=val_steps,
                       workers=0,
                       #verbose=2,
                       #callbacks=[on_epoch_end,
                       #           save_callback]
                       )

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
        with open("saved_models/padded_resnet_distr_imagenet_model_{}.pkl".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), "wb") as f:
            pickle.dump({'weights': var_dict,
                         "losses": self.all_losses,
                         "val_acc": self.accs}, f)

    def load(self, filename):
        import pickle
        with open(filename, "rb") as f:
            model_dict = pickle.load(f)
        assign_ops = []
        for var in tf.global_variables():
            if 'model' in var.name:
                assign_ops.append(var.assign(model_dict['weights'][var.name]))
        sess = tf.get_default_session()
        sess.run(assign_ops)
        return model_dict


if __name__ == "__main__":
    model = ResnetModel()
    model.train()

