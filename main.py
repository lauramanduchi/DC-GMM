import time
import argparse
from pathlib import Path
import yaml
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import uuid
from ast import literal_eval as make_tuple
from heart_echo.numpy import HeartEchoDataset
from heart_echo.Helpers import LABELTYPE
from UTKFace.numpy import UTKFaceDataLoader
from UTKFace.utils.labels import Label
import math

from source.data import DataGenerator
from source.data import DataGenerator_utkface
from projector_plugin import ProjectorPlugin

import cv2

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

import source.utils as utils
from source.model import DCGMM

# project-wide constants:
ROOT_LOGGER_STR = "DCGMM"
LOGGER_RESULT_FILE = "logs.txt"
CHECKPOINT_PATH = 'models'  # "autoencoder/cp.ckpt"

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

HE_VIEWS = ["LA", "KAKL", "KAPAP", "KAAP", "CV"]

# Singleton loader
utkface_loader = None


def get_data(args, configs):
    if args.data == 'MNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train / 255.
        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = x_test / 255.
        x_test = np.reshape(x_test, (-1, 28 * 28))

    elif args.data == 'fMNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train / 255.
        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = x_test / 255.
        x_test = np.reshape(x_test, (-1, 28 * 28))

    elif args.data == 'Reuters':
        file_train = "dataset/reuters/reutersidf10k_train.npy"
        file_test = "dataset/reuters/reutersidf10k_test.npy"
        rtk10k_train = np.load(file_train, allow_pickle=True).item()
        rtk10k_test = np.load(file_test, allow_pickle=True).item()
        x_train = rtk10k_train['data']
        y_train = rtk10k_train['label']
        x_test = rtk10k_test['data']
        y_test = rtk10k_test['label']

    elif args.data == 'heart_echo':
        if configs['data']['label'] == "view":
            he_label_type = LABELTYPE.VIEW
        elif configs['data']['label'] == "preterm_categorical":
            he_label_type = LABELTYPE.PRETERM_CATEGORICAL
        elif configs['data']['label'] == "preterm_categorical_alternate":
            he_label_type = LABELTYPE.PRETERM_CATEGORICAL_ALTERNATE
        else:
            raise NotImplementedError("Unknown label type {}".format(configs['data']['label']))

        he_train_dataset = HeartEchoDataset(
            [67, 55, 88, 93, 74, 87, 95, 83, 40, 39, 61, 60, 80, 103, 104, 84, 53, 68, 73, 70, 66, 82, 85, 105, 57, 69,
             98, 56, 63, 71, 42, 75, 37, 62, 54, 102, 92, 33, 50, 78], scale_factor=0.25,
            label_type=he_label_type, resize=(64, 64), frame_block_size=1)

        x_train, y_train = he_train_dataset.get_data()
        x_train = x_train / 255.

        he_test_dataset = HeartEchoDataset(
            [34, 41, 59, 99, 36, 48, 101, 89, 81, 94, 86, 97, 72, 100, 38, 49, 90, 91, 30, 58, 96, 31, 35, 47, 79],
            scale_factor=0.25, label_type=he_label_type, resize=(64, 64), frame_block_size=1)

        x_test, y_test = he_test_dataset.get_data()
        x_test = x_test / 255.

        # Reshape
        if configs['data']['flatten'] == "yes":
            x_train = x_train.reshape(-1, 4096)
            x_test = x_test.reshape(-1, 4096)
        else:
            x_train = x_train.reshape(-1, 64, 64, 1)
            x_test = x_test.reshape(-1, 64, 64, 1)

        # Map labels to ints for views
        if configs['data']['label'] == "view":
            y_train = np.array(list(map(HE_VIEWS.index, y_train)))
            y_test = np.array(list(map(HE_VIEWS.index, y_test)))

        # Convert onehot for preterm categorical
        elif configs['data']['label'] in ["preterm_categorical", "preterm_categorical_alternate"]:
            y_train = np.array([np.where(r == 1)[0][0] for r in y_train])
            y_test = np.array([np.where(r == 1)[0][0] for r in y_test])

    elif args.data == 'stl10':
        X = np.load("dataset/stl10/stl_features.npy")
        X = X.astype('float32')
        y = np.load("dataset/stl10/stl_label.npy")
        y = y.astype('int32')
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


    elif args.data == 'utkface':
        global utkface_loader

        if utkface_loader is None:
            if configs['data']['label'] == "age":
                label_type = Label.Type.AGE
            elif configs['data']['label'] == "age_bins":
                label_type = Label.Type.AGE_BINS_UNIFORM
            elif configs['data']['label'] == "age_bins_manual":
                label_type = Label.Type.AGE_BINS_MANUAL
            elif configs['data']['label'] == "gender":
                label_type = Label.Type.GENDER
            elif configs['data']['label'] == "ethnicity":
                label_type = Label.Type.ETHNICITY
            else:
                label_type = Label.Type.NONE

            if label_type == Label.Type.ETHNICITY:
                utkface_loader = UTKFaceDataLoader(label=label_type, filter_age=(18, 50), filter_ethnicity=[0, 1, 2, 3],
                                                   resize=(64, 64))
            elif label_type != Label.Type.AGE_BINS_UNIFORM:
                utkface_loader = UTKFaceDataLoader(label=label_type, filter_age=None, resize=(64, 64))
            else:
                num_bins = configs['training']['num_clusters']
                utkface_loader = UTKFaceDataLoader(label=label_type, filter_age=None, num_age_bins=num_bins,
                                                   resize=(64, 64))

        X, y = utkface_loader.get_data()

        X = X / 255.

        if configs['data']['flatten'] == "yes":
            X = X.reshape(-1, 64*64*3)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    
    return x_train, x_test, y_train, y_test


def loss_DCGMM_mnist(inp, x_decoded_mean):
    x = inp
    loss = 784 * tf.keras.losses.BinaryCrossentropy()(x, x_decoded_mean)
    return loss


def loss_DCGMM_reuters(inp, x_decoded_mean):
    x = inp
    loss = 2000 * tf.keras.losses.MeanSquaredError()(x, x_decoded_mean)
    return loss

def loss_DCGMM_heart_echo(inp, x_decoded_mean):
    x = inp
    loss = 4096 * tf.keras.losses.BinaryCrossentropy()(x, x_decoded_mean)
    return loss


def loss_DCGMM_stl10(inp, x_decoded_mean):
    x = inp
    loss = 2048 * tf.keras.losses.MeanSquaredError()(x, x_decoded_mean)  # 2304
    return loss


def loss_DCGMM_utkface(inp, x_decoded_mean):
    x = inp
    #loss = 12288 * tf.keras.losses.MeanSquaredError()(x, x_decoded_mean)
    loss = 12288 * tf.keras.losses.BinaryCrossentropy()(x, x_decoded_mean)
    return loss


def accuracy_metric(inp, p_c_z):
    y = inp
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(utils.cluster_acc, [y, y_pred], tf.float64)


def pretrain(model, args, ex_name, configs):
    input_shape = configs['training']['inp_shape']
    num_clusters = configs['training']['num_clusters']

    if configs['data']['data_name'] in ["heart_echo", "cifar10", "utkface"]:
        if configs['training']['type'] in ["CNN", "VGG"]:
            input_shape = make_tuple(input_shape)

    # Get the AE from the model
    input = tfkl.Input(shape=input_shape)

    if configs['training']['type'] == "FC":
        f = tfkl.Flatten()(input)
        e1 = model.encoder.dense1(f)
        e2 = model.encoder.dense2(e1)
        e3 = model.encoder.dense3(e2)
        z = model.encoder.mu(e3)
        d1 = model.decoder.dense1(z)
        d2 = model.decoder.dense2(d1)
        d3 = model.decoder.dense3(d2)
        dec = model.decoder.dense4(d3)
    elif configs['training']['type'] == "CNN":
        e1 = model.encoder.conv1(input)
        e2 = model.encoder.conv2(e1)
        f = tfkl.Flatten()(e2)
        z = model.encoder.mu(f)
        d1 = model.decoder.dense(z)
        d2 = model.decoder.reshape(d1)
        d3 = model.decoder.convT1(d2)
        d4 = model.decoder.convT2(d3)
        d5 = model.decoder.convT3(d4)
        dec = tf.sigmoid(d5)
    elif configs['training']['type'] == "VGG":
        enc = input
        for block in model.encoder.layers:
            enc = block(enc)
        f = tfkl.Flatten()(enc)
        z = model.encoder.mu(f)
        d_dense = model.decoder.dense(z)
        d_reshape = model.decoder.reshape(d_dense)
        dec = d_reshape
        for block in model.decoder.layers:
            dec = block(dec)
        dec = model.decoder.convT(dec)
        dec = tf.sigmoid(dec)

    autoencoder = tfk.Model(inputs=input, outputs=dec)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # , decay=args.decay)
    if args.data == 'MNIST' or args.data == 'fMNIST' or args.data == 'heart_echo' or args.data == 'utkface':
        autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy")
    else:
        autoencoder.compile(optimizer=optimizer, loss="mse")
    autoencoder.summary()
    x_train, x_test, y_train, y_test = get_data(args, configs)
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))

    # If the model should be run from scratch:
    if args.pretrain:
        print('\n******************** Pretraining **************************')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="pretrain/autoencoder_tmp/" + ex_name + "/cp.ckpt",
                                                         save_weights_only=True, verbose=1)
        autoencoder.fit(X, X, epochs=args.epochs_pretrain, batch_size=32, callbacks=cp_callback)

        encoder = model.encoder
        input = tfkl.Input(shape=input_shape)
        z, _ = encoder(input)
        z_model = tf.keras.models.Model(inputs=input, outputs=z)
        z = z_model.predict(X)

        estimator = GaussianMixture(n_components=num_clusters, covariance_type='diag', n_init=3)
        estimator.fit(z)
        pickle.dump(estimator, open("pretrain/gmm_tmp/" + ex_name + "_gmm_save.sav", 'wb'))

        print('\n******************** Pretraining Done**************************')
    else:
        if args.data == 'MNIST':
            autoencoder.load_weights("pretrain/MNIST/autoencoder/cp.ckpt")
            estimator = pickle.load(open("pretrain/MNIST/gmm_save.sav", 'rb'))
            print('\n******************** Loaded MNIST Pretrain Weights **************************')
        elif args.data == 'fMNIST':
            autoencoder.load_weights("pretrain/fMNIST/autoencoder/cp.ckpt")
            estimator = pickle.load(open("pretrain/fMNIST/gmm_save.sav", 'rb'))
            print('\n******************** Loaded fMNIST Pretrain Weights **************************')
        elif args.data == 'Reuters':
            autoencoder.load_weights("pretrain/Reuters/autoencoder/cp.ckpt")
            estimator = pickle.load(open("pretrain/Reuters/gmm_save.sav", 'rb'))
        elif args.data =="utkface":
            autoencoder.load_weights("pretrain/autoencoder_tmp/20210129-182822_3ff9f/cp.ckpt")
            estimator = pickle.load(open("pretrain/gmm_tmp/20210129-182822_3ff9f_gmm_save.sav", 'rb'))
        elif args.data =="stl10":
            autoencoder.load_weights("pretrain/stl10/autoencoder/cp.ckpt")
            estimator = pickle.load(open("pretrain/stl10/gmm_save.sav", 'rb'))
        elif args.data == "heart_echo":
            autoencoder.load_weights("pretrain/autoencoder_tmp/20210203-141727_c1f28/cp.ckpt")
            estimator = pickle.load(open("pretrain/gmm_tmp/20210203-141727_c1f28_gmm_save.sav", 'rb'))
        else:
            print('\nPretrained weights for {} not available, please rerun with \'--pretrain True option\''.format(
                args.data))
            exit(1)
    
    encoder = model.encoder
    input = tfkl.Input(shape=input_shape)
    z, _ = encoder(input)
    z_model = tf.keras.models.Model(inputs=input, outputs=z)

    # Assign weights to GMM mixtures of DCGMM
    mu_samples = estimator.means_
    sigma_samples = estimator.covariances_
    model.c_mu.assign(mu_samples)
    model.log_c_sigma.assign(np.log(sigma_samples))

    yy = estimator.predict(z_model.predict(X))
    acc = utils.cluster_acc(yy, Y)
    pretrain_acc = acc
    print('\nPretrain accuracy: ' + str(acc))

    return model, pretrain_acc


def run_experiment(args, configs, loss):
    # Set paths
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    experiment_path = args.results_dir / configs['data']['data_name'] / ex_name
    experiment_path.mkdir(parents=True)

    x_train, x_test, y_train, y_test = get_data(args, configs)

    acc_tot = []
    nmi_tot = []
    ari_tot = []

    for i in range(args.runs):
        model = DCGMM(**configs['training'])

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, decay=args.decay)

        def learning_rate_scheduler(epoch):
            initial_lrate = args.lr
            drop = args.decay_rate
            epochs_drop = args.epochs_lr
            lrate = initial_lrate * math.pow(drop,
                                             math.floor((1 + epoch) / epochs_drop))
            return lrate

        if args.lrs:
            cp_callback = [tf.keras.callbacks.TensorBoard(log_dir='logs/' + ex_name),
                           tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)]
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

        elif args.save_model:
            checkpoint_path = CHECKPOINT_PATH + '/' + configs['data']['data_name'] + '/' + ex_name
            cp_callback = [tf.keras.callbacks.TensorBoard(log_dir='logs/' + ex_name),
                           tf.keras.callbacks.ModelCheckpoint(
                               filepath=checkpoint_path,
                               verbose=1,
                               save_weights_only=True,
                               period=100)]
        else:
            cp_callback = [tf.keras.callbacks.TensorBoard(log_dir='logs/' + ex_name)]

        model.compile(optimizer, loss={"output_1": loss}, metrics={"output_4": accuracy_metric})

        # pretrain model
        model, pretrain_acc = pretrain(model, args, ex_name, configs)

        if args.q > 0:
            alpha = 1000 * np.log((1 - args.q) / args.q)
        else:
            alpha = args.alpha

        # create data generators
        if args.data == 'utkface':
            gen = DataGenerator_utkface(x_train, y_train, num_constrains=args.num_constrains, alpha=alpha, q=args.q,
                                        batch_size=args.batch_size, ml=args.ml)
            test_gen = DataGenerator_utkface(x_test, y_test, batch_size=args.batch_size).gen()

        else:
            gen = DataGenerator(x_train, y_train, num_constrains=args.num_constrains, alpha=alpha, q=args.q,
                                  batch_size=args.batch_size, ml=args.ml)
            test_gen = DataGenerator(x_test, y_test, batch_size=args.batch_size).gen()

        train_gen = gen.gen()
        
        # fit model
        model.fit(train_gen, validation_data=test_gen, steps_per_epoch=int(len(y_train)/args.batch_size), validation_steps=len(y_test)//args.batch_size, epochs=args.num_epochs, callbacks=cp_callback)

        # results
        rec, z_sample, p_z_c, p_c_z = model.predict([x_train, np.zeros(len(x_train))])
        yy = np.argmax(p_c_z, axis=-1)
        acc = utils.cluster_acc(y_train, yy)
        nmi = normalized_mutual_info_score(y_train, yy)
        ari = adjusted_rand_score(y_train, yy)
        ml_ind1 = gen.ml_ind1
        ml_ind2 = gen.ml_ind2
        cl_ind1 = gen.cl_ind1
        cl_ind2 = gen.cl_ind2
        count = 0
        if args.num_constrains == 0:
            sc = 0
        else:
            maxx = len(ml_ind1) + len(cl_ind1)
            for i in range(len(ml_ind1)):
                if yy[ml_ind1[i]] == yy[ml_ind2[i]]:
                    count += 1
            for i in range(len(cl_ind1)):
                if yy[cl_ind1[i]] != yy[cl_ind2[i]]:
                    count += 1
            sc = count / maxx


        if args.data == 'MNIST':
            f = open("results_MNIST.txt", "a+")
        elif args.data == 'fMNIST':
            f = open("results_fMNIST.txt", "a+")
        elif args.data == 'Reuters':
            f = open("results_reuters.txt", "a+")
        elif args.data == 'heart_echo':
            f = open("results_heart_echo.txt", "a+")
            f.write("%s, %s. " % (configs['data']['label'], configs['training']['type']))
        elif args.data == 'stl10':
            f = open("results_stl.txt", "a+")
        elif args.data == 'utkface':
            f = open("results_utkface.txt", "a+")
            f.write("%s. " % (configs['data']['label']))
        f.write("Epochs= %d, num_constrains= %d, ml= %d, alpha= %d, batch_size= %d, learning_rate= %f, q= %f, "
                "pretrain_e= %d, gen_old=  %d, "
                % (args.num_epochs, args.num_constrains, args.ml, alpha, args.batch_size, args.lr, args.q,
                   args.epochs_pretrain, args.gen_old))

        if args.lrs == True:
            f.write("decay_rate= %f, epochs_lr= %d, name= %s. " % (args.decay_rate, args.epochs_lr, ex_name))
        else:
            f.write("decay= %f, name= %s. " % (args.decay, ex_name))

        f.write("Pretrain accuracy: %f , " % (pretrain_acc))
        f.write("Accuracy train: %f, NMI: %f, ARI: %f, sc: %f.\n" % (acc, nmi, ari, sc))

        rec, z_sample, p_z_c, p_c_z = model.predict([x_test, np.zeros(len(x_test))])
        yy = np.argmax(p_c_z, axis=-1)
        acc = utils.cluster_acc(y_test, yy)
        nmi = normalized_mutual_info_score(y_test, yy)
        ari = adjusted_rand_score(y_test, yy)

        acc_tot.append(acc)
        nmi_tot.append(nmi)
        ari_tot.append(ari)

        f.write("Accuracy test: %f, NMI: %f, ARI: %f.\n" % (acc, nmi, ari))
        f.close()
        print(str(acc))
        print(str(nmi))
        print(str(ari))

        # Save confusion matrix
        conf_mat = utils.make_confusion_matrix(y_test, yy, configs['training']['num_clusters'])
        np.save("logs/" + ex_name + "/conf_mat.npy", conf_mat)

        # Save embeddings
        if args.save_embedding:
            proj = ProjectorPlugin("logs/" + ex_name, z_sample)

            if args.data == 'heart_echo':
                if configs['data']['label'] == "view":
                    proj.save_labels([HE_VIEWS[label] for label in y_test])

                else:
                    proj.save_labels(y_test)

            else:
                proj.save_labels(y_test)

            # Add images to projector
            if args.data == 'heart_echo':
                proj.save_image_sprites(x_test, 64, 64, 1)
            elif args.data == 'MNIST':
                proj.save_image_sprites(x_test, 28, 28, 1, True)
            elif args.data == 'utkface':
                proj.save_image_sprites(x_test, 64, 64, 3)

            proj.finalize()

    if args.runs > 1:

        acc_tot = np.array(acc_tot)
        nmi_tot = np.array(nmi_tot)
        ari_tot = np.array(ari_tot)

        if args.data == 'MNIST':
            f = open("evaluation_MNIST.txt", "a+")
        elif args.data == 'fMNIST':
            f = open("evaluation_fMNIST.txt", "a+")
        elif args.data == 'Reuters':
            f = open("evaluation_reuters.txt", "a+")
        elif args.data == 'har':
            f = open("evaluation_har.txt", "a+")
        elif args.data == 'cifar10':
            f = open("evaluation_cifar.txt", "a+")
        elif args.data == 'heart_echo':
            f = open("evaluation_heart_echo.txt", "a+")
        elif args.data == 'stl10':
            f = open("evaluation_stl.txt", "a+")
        elif args.data == 'utkface':
            f = open("evaluation_utkface.txt", "a+")

        f.write("Epochs= %d, num_constrains= %d, ml= %d, alpha= %d, batch_size= %d, learning_rate= %f, q= %f, "
                "pretrain_e= %d, gen_old=  %d, "
                % (args.num_epochs, args.num_constrains, args.ml, alpha, args.batch_size, args.lr, args.q,
                   args.epochs_pretrain, args.gen_old))
        if args.lrs == True:
            f.write("decay_rate= %f, epochs_lr= %d, runs= %d, name= %s. " % (args.decay_rate, args.epochs_lr, args.runs, ex_name))
        else:
            f.write(
            "decay= %f, runs= %d, name= %s. "
            % (args.decay, args.runs, ex_name))

        f.write("Pretrain accuracy: %f , " % (pretrain_acc))
        f.write("Accuracy: %f std %f, NMI: %f std %f, ARI: %f std %f. \n" % (
            np.mean(acc_tot), np.std(acc_tot), np.mean(nmi_tot), np.std(nmi_tot), np.mean(ari_tot), np.std(ari_tot)))


def main():
    project_dir = Path(__file__).absolute().parent

    parser = argparse.ArgumentParser()

    # parameters of the model
    parser.add_argument('--data',
                        default='MNIST',
                        type=str,
                        choices=['MNIST', 'fMNIST', 'Reuters', 'stl10', 'heart_echo', 'utkface'],
                        help='specify the data (MNIST, fMNIST, Reuters, stl10, heart_echo, utkface)')
    parser.add_argument('--num_epochs',
                        default=1000,
                        type=int,
                        help='specify the number of epochs')
    parser.add_argument('--num_constrains',
                        default=6000,
                        type=int,
                        help='specify the number of constrains')
    parser.add_argument('--batch_size',
                        default=256,
                        type=int,
                        help='specify the batch size')
    parser.add_argument('--alpha',
                        default=10000,
                        type=int,
                        help='specify alpha, the weight importance of the constraints (higher means higher confidence)')
    parser.add_argument('--q',
                        default=0,
                        type=float,
                        help='specify the flip probability of the labels')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='specify learning rate')
    parser.add_argument('--decay',
                        default=0.00001,
                        type=float,
                        help='specify decay')
    parser.add_argument('--ml',
                        default=0,
                        type=int,
                        choices=[0, 1, -1],
                        help='0: random choice, 1: only must-link, -1: only cannot-link')
    parser.add_argument('--w',
                        default=1,
                        type=float,
                        help='w')
    parser.add_argument('--decay_rate',
                        default=0.9,
                        type=float,
                        help='specify decay')
    parser.add_argument('--epochs_lr',
                        default=20,
                        type=int,
                        help='specify decay')
    parser.add_argument('--lrs',
                        default=True,
                        type=bool,
                        help='specify decay')

    # other parameters
    parser.add_argument('--runs',
                        default=1,
                        type=int,
                        help='number of runs, the results will be averaged')
    parser.add_argument('--results_dir',
                        default=project_dir / 'experiments',
                        type=lambda p: Path(p).absolute(),
                        help='specify the folder where the results get saved')
    parser.add_argument('--pretrain', default=False, type=bool,
                        help='True to pretrain the autoencoder, False to use pretrained weights')
    parser.add_argument('--epochs_pretrain', default=10, type=int,
                        help='Specify the number of pre-training epochs')
    parser.add_argument('--save_model', default=False, type=bool,
                        help='True to save the model')
    parser.add_argument('--gen_old', default=False, type=bool,
                        help='True to use old generator')
    parser.add_argument("--save_embedding", default=False, type=bool, help="Save embedded representation")
    parser.add_argument("--cluster", default=False, type=bool, help="Script running on cluster")
    parser.add_argument("--ex_name", default="", type=str, help="Specify experiment name")
    parser.add_argument("--config_override", default="", type=str, help="Specify config.yml override file")

    args = parser.parse_args()

    if args.data == "MNIST" or args.data == "fMNIST":
        config_path = project_dir / 'configs' / 'MNIST.yml'
        loss = loss_DCGMM_mnist
    elif args.data == "Reuters":
        config_path = project_dir / 'configs' / 'Reuters.yml'
        loss = loss_DCGMM_reuters
    elif args.data == "stl10":
        config_path = project_dir / 'configs' / 'stl10.yml'
        loss = loss_DCGMM_stl10
    elif args.data == "heart_echo":
        config_path = project_dir / 'configs' / 'heart_echo.yml'
        loss = loss_DCGMM_heart_echo
    elif args.data == "utkface":
        config_path = project_dir / 'configs' / 'utkface.yml'
        loss = loss_DCGMM_utkface

    # Check for config override
    if args.config_override is not "":
        config_path = Path(args.config_override)

    with config_path.open(mode='r') as yamlfile:
        configs = yaml.safe_load(yamlfile)

    run_experiment(args, configs, loss)


if __name__ == "__main__":
    main()
