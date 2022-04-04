from __future__ import absolute_import
from __future__ import print_function
from imblearn.over_sampling import SMOTE

import numpy as np
import argparse
import os
import imp
import re
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger
from focal_loss import BinaryFocalLoss

import tsaug

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--cbloss', type=str, help='Use Class-Balanced Loss',
                    default=False)
parser.add_argument('--beta', type=str, help='beta for CBL',
                    default=0.9)
parser.add_argument('--smote', type=str, help='oversample data',
                    default=False)
parser.add_argument('--timewarp', type=str, help='oversample data',
                    default=False)
parser.add_argument('--addnoise', type=str, help='oversample data',
                    default=False)
parser.add_argument('--focal_loss', type=str, help='Use Class-Balanced Focal Loss',
                    default=False)
parser.add_argument('--gamma', type=str, help='Gamma value for Class-Balanced Focal Loss',
                    default=False)                        
args = parser.parse_args()
print(args)

if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='normal_value',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]


#Setup the normalization strat
normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str_previous.start_time_zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl


# Read data
print('==> reading data')
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)
X=train_raw[0]
y=train_raw[1]

# Oversample
if args.smote:
    print("Oversampling with SMOTE\n")
    sm = SMOTE(sampling_strategy='minority',random_state=42)

    #First, reshape our data with shape (samples, time, fts) to (samples, fts)
    n_fts=np.shape(X)[2]
    n_samples=np.shape(X)[0]
    # X_reshape=np.reshape(X,(X.shape[0],X.shape[1]))
    X_res=[]
    y_res=[]
    for t in range(0,X.shape[1]):
        X_res_t, y_res_t = sm.fit_resample(X[:,t,:], y)
        X_res.append(X_res_t)
        y_res.append(y_res_t)

    #Reshape to (samples, time, fts)
    arr=np.vstack(X_res)
    X=np.reshape(arr,(np.shape(X_res)[1],np.shape(X_res)[0],np.shape(X_res)[2]))

    #Now we have labels in the shape of time, samples because we used SMOTE 
    #for each time step, but every timestep has the same label i.e. columns of 1 
    #and 0
    y=np.vstack(y_res)[1,:]

if args.timewarp:
    #Determine number of samples to add so we get a 50/50´balanced set
    print("Oversampling with TimeWarp\n")
    no_min=sum(y)
    no_maj=len(y)-sum(y)
    no_oversamples=no_maj-no_min
    X_transf=np.zeros(shape=(no_oversamples,X.shape[1],X.shape[2])) #shape: samples, time, features
    X_min=X[y==1]
    counter=0
    for sample in range(0,no_oversamples):
        for col in range(0,X_transf[sample,:,:].shape[1]):
            if counter==X_min.shape[0]:
                counter=0
                X_transf[sample,:,col]=np.round(tsaug.TimeWarp().augment(X_min[counter,:,col]))
                counter+=1

    X=np.concatenate((X,X_transf))
    y=np.concatenate((y,np.ones(no_oversamples)))

if args.addnoise:
    #Determine number of samples to add so we get a 50/50´balanced set
    print("Oversampling with AddNoise\n")
    no_min=sum(y)
    no_maj=len(y)-sum(y)
    no_oversamples=no_maj-no_min
    X_transf=np.zeros(shape=(no_oversamples,X.shape[1],X.shape[2])) #shape: samples, time, features
    X_min=X[y==1]
    counter=0
    for sample in range(0,no_oversamples):
        for col in range(0,X_transf[sample,:,:].shape[1]):
            if counter==X_min.shape[0]:
                counter=0
                X_transf[sample,:,col]=np.round(tsaug.AddNoise(normalize=True, kind='multiplicative').augment(X_min[counter,:,col]))
                counter+=1

    X=np.concatenate((X,X_transf))
    y=np.concatenate((y,np.ones(no_oversamples)))

# print('trainraw\n')
# print(train_raw)
# print(np.array(train_raw[1]).T)
# train_dataset = tf.data.Dataset.from_tensor_slices((train_raw[0], train_raw[1]))
# print('x\n')
# print(x.shape, x.dtype) #14655, 48, 76
# print(x)
# print('y\n')
# print(np.shape(y))  #14655
# # y=np.array(y)
# print(y)

# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}
# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).


if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
elif args.cbloss:
    print('=> using Class Balanced Binary Cross Entropy Loss')
    samples_per_cls=[len(y)-sum(y),sum(y)]
    loss = ['binary_crossentropy'] * 2
    beta=float(args.beta)
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    loss_weights = (1.0-beta)/np.array(effective_num)
    loss_weights = loss_weights / np.sum(loss_weights) * 2
elif args.focal_loss:
    print('=> using Class Balanced Binart Focal Loss')
    samples_per_cls=[len(y)-sum(y),sum(y)]
    if args.gamma:
        gamma=args.gamma
    else:
        gamma=2
    loss = BinaryFocalLoss(gamma=gamma)
    beta=float(args.beta)
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    loss_weights = (1.0-beta)/np.array(effective_num)
    loss_weights = loss_weights / np.sum(loss_weights) * 2
else:
    print('=> using Binary Cross Entropy Loss')
    loss = 'binary_crossentropy'
    loss_weights = None

print('Loss:',loss,'\n')

optimizer=tf.keras.optimizers.Adam(lr=args.lr, beta_1=args.beta_1)

model.compile(optimizer=optimizer,
              loss=loss,
              loss_weights=loss_weights)
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))




if target_repl:
    T = train_raw[0][0].shape[0]

    def extend_labels(data):
        data = list(data)
        labels = np.array(data[1])  # (B,)
        data[1] = [labels, None]
        data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
        data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
        return data

    train_raw = extend_labels(train_raw)
    val_raw = extend_labels(val_raw)

if args.mode == 'train':

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')
    
    class_weights={0:1, 1:2}

    model.fit(x=X,
              y=y,
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")
