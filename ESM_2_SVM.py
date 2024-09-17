import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.svm import SVC
import pickle
import torch
import glob

path = 'trainingmodellarge_0620_full_L6/*.*'
files = glob.glob(path)
X = []
Y = []
for name in files:
    if 'NTF' in name:
        Y.append(0)
    else:
        Y.append(1)
    with open(name) as f:
     embedding = torch.load(name)
     embedding = embedding['representations'][6].numpy()
     #print(len(embedding))
     #print(len(embedding[0]))
     if len(embedding) < 1022:
         padding = np.zeros(((1022-len(embedding)), 320))
         embedding = embedding.tolist() + padding.tolist()
         embedding = np.array(embedding)
     #print(len(embedding[0]))
     #print(len(embedding))
     X.append(embedding)

print(len(X))
print(len(X[0]))

X = np.array(X)
Y = np.array(Y)
np.save()
np.save()
emb_dim = len(X[0])

print("emb dim: ", emb_dim)
print("y:", len(Y))
print("X:", len(X))
print(np.shape(X))

print("NTF: ", np.count_nonzero(Y == 0))
print("TF: ", np.count_nonzero(Y == 1))

#X = np.reshape(X, (len(Y), X.shape[0]* X.shape[1]))  # fit input must be 2d

# create multi balanced dataset, train multi classifier then average?
#X = X[1400:]
#Y = Y[1400:]

# oversample. It randomly samples the minority class until the numbers is equal to majority class.
#from imblearn.over_sampling import RandomOverSampler
#rus = RandomOverSampler(random_state=42)
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
weights = {0:0.5, 1:0.5}

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)
# Define a voting function
def ensemble_voting(predictions1, predictions2):
    # Compute the average of the predicted probabilities from both models
    ensemble_predictions = (predictions1 + predictions2) / 2
    # Convert probabilities to binary predictions (0 or 1) based on a threshold
    threshold = 0.5
    ensemble_predictions_binary = np.where(ensemble_predictions >= threshold, 1, 0)
    return ensemble_predictions_binary

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
frac = 0.3
k_num = 0
for train, test in kfold.split(X, Y):
    k_num += 1
    print("******************", k_num, "******************")
    x_train = X[train]
    x_test = X[test]
    y_train = Y[train]
    y_test = Y[test]

    # downsample the NTF
    """
    label_NTF_indices = np.where(y_train == 0)[0]
    x_train_NTF, y_train_NTF = resample(x_train[label_NTF_indices], y_train[label_NTF_indices], replace=True,n_samples=int(x_train[label_NTF_indices].shape[0] * frac), random_state=123)
    label_TF_indices = np.where(y_train == 1)[0]
    x_train_TF, y_train_TF = resample(x_train[label_TF_indices], y_train[label_TF_indices], replace=False,n_samples=len(label_TF_indices), random_state=123)
    x_train = np.concatenate([x_train_TF, x_train_NTF])
    y_train = np.concatenate([y_train_TF, y_train_NTF])
    """

    #x_train, y_train = rus.fit_resample(x_train, y_train) #RandomOverSampler
    x_train_u, y_train_u = rus.fit_resample(x_train, y_train)  # RandomOverSampler

    sm = SMOTE()
    x_train_s, y_train_s = sm.fit_resample(x_train, y_train)  # SMOTE oversampling
    #ada = ADASYN(random_state=42)
    #x_train, y_train = ada.fit_resample(x_train, y_train)

    #print("After sampling NTF: ", np.count_nonzero(y_train == 0))
    #print("After sampling TF: ", np.count_nonzero(y_train == 1))

    # Support vector classifier
    #model = SVC(C=1.0, kernel='linear', class_weight='balanced').fit(x_train_s, y_train_s)
    #model2 = SVC(C=1.0, kernel='linear', class_weight='balanced').fit(x_train_u, y_train_u)
    model = SVC(kernel='linear',  class_weight='balanced', random_state=44, probability=True, verbose=True).fit(x_train, y_train) #class_weight='balanced', rbf
    #model = DecisionTreeClassifier(random_state=42).fit(x_train , y_train)
    #model = GradientBoostingClassifier(random_state=42).fit(x_train , y_train)
    #model = RandomForestClassifier(random_state=42).fit(x_train , y_train)#max_depth=9,max_features="log2",max_leaf_nodes=9,n_estimators=25,
    #model = LogisticRegression(solver='lbfgs').fit(x_train , y_train)
    #model = xgb.XGBClassifier(objective="binary:logistic", random_state=42).fit(x_train_s , y_train_s)
    #model = ensemble.EasyEnsembleClassifier(random_state=42).fit(x_train, y_train)

    model_name = 'saved_model/SVM/svm_model_t33_0627' + str(k_num) + '.pkl'
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)

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
