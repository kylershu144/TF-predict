import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from random import sample
import tensorflow as tf
from keras import layers
import keras
import math
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, LSTM, Activation, Permute, Multiply, Lambda, Attention, Embedding,Bidirectional
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import resample
from keras import regularizers
import os
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

#only cnn with 1D
l2_reg=0.01
def create_model_1(input_shape, num_classes):
    model = Sequential()
    # Convolutional layers
    model.add(Conv1D(512, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(256, kernel_size=3, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # Fully connected layers
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))) #
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_reg))) #
    return model

# can acheive 94%
def create_model_2(input_shape, num_classes):
    model = tf.keras.Sequential([
        layers.Conv1D(512, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        #layers.Dropout(0.5),

        layers.Conv1D(256, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        #layers.Dropout(0.5),

        layers.Flatten(),

        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])
    return model

#CNN + LSTM + attention
def create_model_3 (input_shape, num_classes):
    inputs = Input(shape=input_shape)
    conv1 = Conv1D(128, kernel_size=3, activation='relu')(inputs)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(bn1)
    #drop1 = Dropout(0.2)(pool1)
    conv2 = Conv1D(256, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(bn2)
    conv3 = Conv1D(512, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(pool2)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(bn3)
    # LSTM layer
    lstm = LSTM(256, return_sequences=True)(pool3)
    # Attention layer
    attention = Attention()([lstm, lstm])
    # Apply attention weights
    attended_lstm = Multiply()([lstm, attention])
    attended_lstm = Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended_lstm)
    flatten = Flatten()(attended_lstm)
    #dense1 = Dense(256, activation='relu')(flatten)
    #dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(256, activation='relu')(flatten)
    dropout2 = Dropout(0.2)(dense2)
    output = Dense(num_classes, activation='sigmoid')(dropout2)
    model = Model(inputs=inputs, outputs=output)
    return model

# BiDirectional LSTM
def create_model_4(input_shape,num_classes):
    inp = Input(shape=input_shape)
    #x = Embedding(input_dim=22, output_dim=128)(inp)
    '''
    Here 64 is the size(dim) of the hidden state vector as well as the output vector. Keeping return_sequence we want the output for the entire sequence. So what is the dimension of output for this layer?
        64*70(maxlen)*2(bidirection concat)
    CuDNNLSTM is fast implementation of LSTM layer in Keras which only runs on GPU
    '''
    x = Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64, return_sequences=True))(inp)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(num_classes, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    return model

#transformer

# Define the TextCNN model
def create_model_5(input_shape, num_classes):
    vocabulary_size = 22
    embedding_dim = 128
    num_filters = 64
    filter_sizes = [3, 4, 5]
    # Input layer
    input_layer = Input(shape=input_shape)
    # Embedding layer
    #embedding_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)(input_layer)
    # Convolutional and pooling layers
    pooled_outputs = []
    for filter_size in filter_sizes:
        conv_layer = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(input_layer)
        pool_layer = GlobalMaxPooling1D()(conv_layer)
        pooled_outputs.append(pool_layer)
    pooled_outputs_concat = tf.concat(pooled_outputs, axis=1)
    # Fully connected layer
    dense_layer = Dense(units=256, activation='relu')(pooled_outputs_concat)
    # Output layer
    output_layer = Dense(units=num_classes, activation='sigmoid')(dense_layer)
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

#simple model
def create_model(input_shape, num_classes):
    input = Input(shape=input_shape)

    layer1 = layers.Dense(1024, activation='relu')(input)
    layer2 = layers.Dense(1024, activation='relu')(layer1)
    layerB = layers.Dropout(0.5)(layer2)
    layerA = layers.Dense(512, activation='relu')(layerB)
    layer3 = layers.Dropout(0.5)(layerA)
    layer4 = layers.Flatten()(layer3)

    output_layer = layers.Dense(num_classes, activation='sigmoid')(layer4)
    model = Model(inputs=input, outputs=output_layer)
    return model

#CNN + Bidirectional LSTM
def create_model_7(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(256)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    return model

"""
import torch
import glob
import h5py

print("opening data...")
path = 'trainingmodellarge_0620_full_L6/*.*'
files = glob.glob(path)
X = []
Y = []
print("appending data...")
for name in files:
    if 'NTF' in name:
        Y.append(0)
    else:
        Y.append(1)
    with open(name) as f:
     embedding = torch.load(name)
     embedding = embedding['representations'][6].numpy()
     if len(embedding) < 1022:
         padding = np.zeros(((1022-len(embedding)), 320))
         embedding = embedding.tolist() + padding.tolist()
         embedding = np.array(embedding)
     X.append(embedding)

X = np.array(X)
Y = np.array(Y)
"""
X = np.load("X_chunk.npy")
Y = np.load("Yfull.npy")

emb_dim = len(X[0, 0, :])
print("emb dim: ", emb_dim)
print("y:", len(Y))
print("X:", len(X))

print("NTF: ", np.count_nonzero(Y == 0))
print("TF: ", np.count_nonzero(Y == 1))

#X = np.reshape(X, (len(Y), len(X[0]) * len(X[0][0]) * len(X[0][0][0])))  # fit input must be 2d

# create multi balanced dataset, train multi classifier then average?
# oversample. It randomly samples the minority class until the numbers is equal to majority class.
#from imblearn.over_sampling import RandomOverSampler
#rus = RandomOverSampler(random_state=42)
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
sm = SMOTE()
ada = ADASYN(random_state=42)

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

METRICS = [
      #keras.metrics.TruePositives(name='tp'),
      #keras.metrics.FalsePositives(name='fp'),
      #keras.metrics.TrueNegatives(name='tn'),
      #keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      #keras.metrics.AUC(name='auc'),
      #keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

class BinaryF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.f1_score = tf.keras.metrics.F1Score(num_classes=1, threshold=self.threshold)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.greater_equal(y_pred, self.threshold)
        self.f1_score.update_state(y_true, y_pred)
    def reset_states(self):
        self.f1_score.reset_states()
    def result(self):
        return self.f1_score.result()


from keras import backend as K
def balanced_binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # casting y_true to float
    weights = (y_true * 2.) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce
#loss_function = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)  # can down-weight easy examples and focus more on hard example
#loss_function = ohem_loss()
loss_function = keras.losses.BinaryCrossentropy()


batch_size = 32
epochs = 20
lr = 0.001
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True
)
k_num = 0
training_f1_kfold = []
validation_f1_kfold = []
training_specificity_kfold = []
validation_specificity_kfold = []
training_sensitivity_kfold = []
validation_sensitivity_kfold = []
training_accuracy_kfold = []
validation_accuracy_kfold = []
training_NTF = []
training_TF = []
validation_NTF = []
validation_TF = []
frac = 0.40

#X = np.reshape(X, (-1, emb_dim, 1))

for train, test in kfold.split(X, Y):
    k_num+=1
    x_train = X[train]
    x_test = X[test]
    y_train = Y[train]
    y_test = Y[test]
    print("*******************" + str(k_num) + "**********************")
    # downsample the NTF
    """
    label_NTF_indices = np.where(y_train == 0)[0]
    x_train_NTF, y_train_NTF = resample(x_train[label_NTF_indices], y_train[label_NTF_indices],replace=True, n_samples=int(x_train[label_NTF_indices].shape[0]*frac), random_state=123)
    label_TF_indices = np.where(y_train == 1)[0]
    x_train_TF, y_train_TF = resample(x_train[label_TF_indices], y_train[label_TF_indices], replace=False, n_samples=len(label_TF_indices), random_state=123)
    x_train = np.concatenate([x_train_TF, x_train_NTF])
    y_train = np.concatenate([y_train_TF, y_train_NTF])
    """

    # oversample
    orig_shape = x_train.shape
    print("orig_shape:",orig_shape)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    #print("af_shape:", x_train.shape)

    #x_train, y_train = rus.fit_resample(x_train, y_train) #RandomOverSampler
    x_train_u, y_train_u = rus.fit_resample(x_train, y_train)  # RandomunderSampler
    x_train_u = np.reshape(x_train_u, (x_train_u.shape[0], orig_shape[1], orig_shape[2]))
    #y_train_u = tf.keras.utils.to_categorical(y_train_u, 2)
    x_train_s, y_train_s = sm.fit_resample(x_train, y_train)  # SMOTE oversampling
    x_train_s = np.reshape(x_train_s, (x_train_s.shape[0], orig_shape[1], orig_shape[2]))
    #y_train_s = tf.keras.utils.to_categorical(y_train_s, 2)
    #x_train_a, y_train_a = ada.fit_resample(x_train, y_train) # adasyn oversampling
    #x_train_a = np.reshape(x_train_a, (x_train_a.shape[0], orig_shape[1], orig_shape[2]))
    #y_train_a = tf.keras.utils.to_categorical(y_train_a, 2)

    x_train = np.reshape(x_train, orig_shape)



    #print("After sampling NTF: ", np.count_nonzero(y_train == 0))
    #print("After sampling TF: ", np.count_nonzero(y_train == 1))

    #input = Input(shape=(2000, 22))

    # Define the paths to save the best models
    checkpoint_path = 'best_model/best_model' + str(k_num) +'.h5'
    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', mode='max', patience=20, restore_best_weights=True)
    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

    model = create_model((len(X[0, :]), emb_dim), 1)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  # 'sgd',tf.keras.optimizers.Adam(learning_rate=0.001),  #
                  loss=loss_function ,#balanced_binary_crossentropy, #'binary_crossentropy',loss_function
                  # loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[METRICS])#METRICS

    #x_gen = HardExampleMiner(model, x_train_u, y_train_u, batch_size)
    #ohem_callback = OHEMCallback(x_gen)

    history = model.fit(x_train_s, y_train_s, #x_gen,
                        epochs=epochs,
                        verbose=1, #show epoch training process 0, 1, 2
                        validation_data=(x_test, y_test),
                        #callbacks=[ohem_callback],
                        callbacks=[checkpoint],
                        batch_size=batch_size)  # class_weight=weights_dict

    model.save('saved_model/CNN/model_t36_0627_'+ str(k_num))


    y_train_pre = (model.predict(x_train)[:] >= 0.5).astype(bool)
    training_f1 = f1_score(y_train, y_train_pre, average='macro')
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pre).ravel()
    training_specificity = tn / (tn + fp)
    training_sensitivity = tp / (tp + fn)
    training_accuracy = balanced_accuracy_score(y_train, y_train_pre)

    training_f1_kfold.append(training_f1)
    training_specificity_kfold.append(training_specificity)
    training_sensitivity_kfold.append(training_sensitivity)
    training_accuracy_kfold.append(training_accuracy)
    # ************************************************************
    y_test_pre = (model.predict(x_test)[:] >= 0.5).astype(bool)
    validation_f1 = f1_score(y_test, y_test_pre, average='macro')
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pre).ravel()
    validation_specificity = tn / (tn + fp)
    validation_sensitivity = tp / (tp + fn)
    validation_accuracy = balanced_accuracy_score(y_test, y_test_pre)

    validation_f1_kfold.append(validation_f1)
    validation_specificity_kfold.append(validation_specificity)
    validation_sensitivity_kfold.append(validation_sensitivity)
    validation_accuracy_kfold.append(validation_accuracy)

    # Get the unique labels and their counts in the training set
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    train_label_counts = dict(zip(train_labels, train_counts))
    # Get the unique labels and their counts in the test set
    test_labels, test_counts = np.unique(y_test, return_counts=True)
    test_label_counts = dict(zip(test_labels, test_counts))
    # Print the number of samples for each label in the training set
    for label, count in train_label_counts.items():
        if int(label) == 0:
            training_NTF.append(count)
        else:
            training_TF.append(count)
    # Print the number of samples for each label in the test set
    for label, count in test_label_counts.items():
        if int(label) == 0:
            validation_NTF.append(count)
        else:
            validation_TF.append(count)

def Average(lst):
    return sum(lst) / len(lst)

print("training.......................")
print("f1:\t\t\t\t", training_f1_kfold)
print("specificity:\t", training_specificity_kfold)
print("sensitivity:\t", training_sensitivity_kfold)
print("accuracy:\t\t", training_accuracy_kfold, "\n")

print("training average.......................")
print("f1:\t\t\t\t", Average(training_f1_kfold))
print("specificity:\t", Average(training_specificity_kfold))
print("sensitivity:\t", Average(training_sensitivity_kfold))
print("accuracy:\t\t", Average(training_accuracy_kfold), "\n")

print("testing.......................")
print("f1:\t\t\t\t", validation_f1_kfold)
print("specificity:\t", validation_specificity_kfold)
print("sensitivity:\t", validation_sensitivity_kfold)
print("accuracy:\t\t", validation_accuracy_kfold, "\n")

print("testing average.......................")
print("f1:\t\t\t\t", Average(validation_f1_kfold))
print("specificity:\t", Average(validation_specificity_kfold))
print("sensitivity:\t", Average(validation_sensitivity_kfold))
print("accuracy:\t\t", Average(validation_accuracy_kfold), "\n")

print("training sample average.......................")
print(Average(training_NTF))
print(Average(training_TF))

print("testing sample average.......................")
print(Average(validation_NTF))
print(Average(validation_TF))