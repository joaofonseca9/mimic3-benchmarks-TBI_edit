from __future__ import absolute_import
from __future__ import print_function
from imblearn.over_sampling import SMOTE

import numpy as np
import pandas as pd
import pickle
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

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from keras.callbacks import ModelCheckpoint, CSVLogger
from focal_loss import BinaryFocalLoss

import tsaug

from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.feature_selection.relevance import calculate_relevance_table

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--beta', type=float, help='beta for CBL',
                    default=0.9)
parser.add_argument('--smote', type=str, help='oversample data',
                    default=False)
parser.add_argument('--timewarp', type=str, help='oversample data',
                    default=False)
parser.add_argument('--addnoise', type=str, help='oversample data',
                    default=False)
parser.add_argument('--loss_type', type=str, help='Loss-type',
                    default='binary_crossentropy')
parser.add_argument('--gamma', type=float, help='Gamma value for Class-Balanced Focal Loss',
                    default=2.0)
parser.add_argument('--class_weight', type=str, help='Class weights for model fit', default=False)
parser.add_argument('--gridsearch', type=str, help='use Grid Search CV', default=False)
parser.add_argument('--ft_selection', type=str, help='Do feature selection', default=False)                        
parser.add_argument('--config_path', type=str, help='Configuration path of the features', default=os.path.join(os.path.dirname(__file__), 'resources/discretizer_config.json'))   
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
                          store_masks=False,
                          impute_strategy=args.imputation,
                          start_time='zero',
                          config_path=args.config_path)

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
if discretizer_header[3] == 'CO2 (ETCO2':
              discretizer_header[3] = 'CO2 (ETCO2, PCO2, etc.)'
              discretizer_header.remove(' etc.)')
              discretizer_header.remove(' PCO2') 
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


def select_fts(X,y, print_final=True):
  n_samples=X.shape[0]
  n_bins=X.shape[1]
  y_df=np.zeros(n_samples*n_bins)
  for i in range(0,n_samples):
      x=pd.DataFrame(X[i], columns=discretizer_header)
      x['id']=i
      y_df[i*n_bins-48:i*n_bins]=y[i]
      if i==0:
          x_df=x.copy()
      else:
          x_df = pd.concat([x_df,x], axis=0)

  y_df=pd.Series(y_df)
  x_df.index=x_df['id']
  y_df.index=x_df['id']
  x_df=x_df.drop(columns=['id'])
  # print relevant fts
  rel_table=calculate_relevance_table(x_df,y_df)
  if print_final:
    print('=> Relevant Features:', len(rel_table[rel_table.relevant==True]['feature']),'\n', rel_table[rel_table.relevant==True]['feature'])
  
  #Select fts
  X_selected = select_features(x_df, y_df)
  X_list=[]

  for i in X_selected.index.unique():
      X_list.append(np.array(X_selected[X_selected.index==i]))
  X=np.concatenate(X_list).reshape((n_samples,n_bins,X_list[0].shape[1]))
  print('=> shape after feature selection ', X.shape)

  return X, rel_table

if args.ft_selection:
  #feature selection for training set
  X, rel_table = select_fts(X,y,print_final= True)

  #select the same features in the validation set
  val_raw_list=[]
  for i in range(0,len(val_raw[0])):
    x_df=pd.DataFrame(val_raw[0][i], columns=discretizer_header)
    x_df=x_df[rel_table[rel_table.relevant==True]['feature']]
    val_raw_list.append(np.array(x_df))
  X_val=np.concatenate(val_raw_list).reshape((len(val_raw[0]),48,val_raw_list[0].shape[1]))
  val_raw=[X_val,val_raw[1]]
# Oversample
if args.smote:
    print("=> Oversampling with SMOTE\n")
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
elif args.timewarp:
    #Determine number of samples to add so we get a 50/50´balanced set
    print("=> Oversampling with TimeWarp\n")
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
elif args.addnoise:
    #Determine number of samples to add so we get a 50/50´balanced set
    print("=> Oversampling with AddNoise\n")
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
else:
    print("=> No oversampling\n")

# Build the model
def create_model(input_dim=X.shape[2],batch_size=args.batch_size, batch_norm=args.batch_norm, learning_rate=args.lr, 
                depth=args.depth, dim=args.dim, dropout=args.dropout,rec_dropout=args.rec_dropout,
                l1=args.l1, l2=args.l2, target_repl_coef=args.target_repl_coef,beta=args.beta, 
                gamma=args.gamma, optimizer=args.optimizer, beta_1=args.beta_1, loss_type=args.loss_type, task='ihm'):
  print("==> using model {}".format(args.network))
  model_module = imp.load_source(os.path.basename(args.network), args.network)
  model = model_module.Network(input_dim=input_dim,dim=dim, batch_norm=batch_norm, dropout=dropout, rec_dropout=rec_dropout, target_repl_coef=target_repl_coef, depth=depth, task=task)   
  suffix = ".bs{}{}{}.ts{}{}".format(batch_size,
                                    ".L1{}".format(l1) if l1 > 0 else "",
                                    ".L2{}".format(l2) if l2 > 0 else "",
                                    args.timestep,
                                    ".trc{}".format(target_repl_coef) if target_repl_coef > 0 else "",
                                    ".trc{}".format(beta) if beta > 0 else "",
                                    ".trc{}".format(gamma) if gamma > 0 else "")
  model.final_name = args.prefix + model.say_name() + suffix
  print("==> model.final_name:", model.final_name)


  # Compile the model
  print("==> compiling the model")
  optimizer_config = {'class_name': optimizer,
                      'config': {'lr': learning_rate,
                                'beta_1': beta_1}}
  # NOTE: one can use binary_crossentropy even for (B, T, C) shape.
  #       It will calculate binary_crossentropies for each class
  #       and then take the mean over axis=-1. Tre results is (B, T).


  if target_repl:
      loss = ['binary_crossentropy'] * 2
      loss_weights = [1 - target_repl_coef, target_repl_coef]
  elif loss_type =='cbloss':
      print('=> using Class Balanced Binary Cross Entropy Loss \n')
      samples_per_cls=[len(y)-sum(y),sum(y)]
      loss = ['binary_crossentropy'] * 2
      beta=float(args.beta)
      effective_num = 1.0 - np.power(beta, samples_per_cls)
      loss_weights = (1.0-beta)/np.array(effective_num)
      loss_weights = loss_weights / np.sum(loss_weights) * 2
  elif loss_type == 'focal_loss':
      print('=> using Class Balanced Binary Focal Loss \n')
      samples_per_cls=[len(y)-sum(y),sum(y)]
      loss = BinaryFocalLoss(gamma=args.gamma)
      beta=float(args.beta)
      effective_num = 1.0 - np.power(beta, samples_per_cls)
      loss_weights = (1.0-beta)/np.array(effective_num)
      loss_weights = loss_weights / np.sum(loss_weights) * 2
  else:
      print('=> using Binary Cross Entropy Loss \n')
      loss = 'binary_crossentropy'
      loss_weights = None

  print('Loss:',loss,'\n')

  optimizer=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1)

  model.compile(optimizer=optimizer,
                loss=loss,
                loss_weights=loss_weights)
  model.summary()

  return model

model=create_model()
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
    train_raw=[X,y]
    val_raw=[X_val,val_raw[1]]
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

    if args.class_weight:
        print("=> using class weights\n")
        if args.class_weight=='proportion':
            class_weights={0: 0.816, 1: 0.184}
        elif args.class_weight=='double':
            class_weights={0: 1, 1: 2}
        else:
            class_weights={0: 0.816, 1: 0.184}
    else:
        class_weights=None

    #GridSearch setup
# batch_size=args.batch_size, batch_norm=args.batch_norm, learning_rate=args.lr, 
#                 depth=args.depth, dim=args.dim, dropout=args.dropout,rec_dropout=args.rec_dropout,
#                 l1=args.l1, l2=args.l2, target_repl_coef=args.target_repl_coef,beta=args.beta, 
#                 gamma=args.gamma, optimizer=args.optimizer, beta_1=args.beta_1, loss_type=args.loss_type, task='ihm'
    if args.gridsearch:
        params = {"learning_rate":[.01],
        "depth":[2],
        "dropout":[0.5],
        "rec_dropout":[0.3],
        "loss_type":['focal_loss','cbloss','binary_crossentropy']}
        
        model_ = KerasClassifier(build_fn = create_model, verbose=0)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(model_, params, scoring='roc_auc', 
                        refit='roc_auc', n_jobs=1, 
                        cv=outer_cv, return_train_score=True )
        
        gs.fit(X, y)
        print("Best Scores: ", gs.best_score_,'\n')
        print("Best Params: ", gs.best_params_,'\n')
        a_file1 = open("cv_scores.pkl", "wb")
        pickle.dump(gs.cv_results_, a_file1)
        a_file1.close()
    else:
        print(X.shape)
        print(y.shape)
        print(val_raw[0].shape)
        model.fit(x=X,
                y=y,
                validation_data=val_raw,
                epochs=n_trained_chunks + args.epochs,
                initial_epoch=n_trained_chunks,
                callbacks=[metrics_callback, saver, csv_logger],
                shuffle=True,
                verbose=args.verbose,
                batch_size=args.batch_size,
                class_weight=class_weights)

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
