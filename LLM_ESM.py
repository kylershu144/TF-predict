import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import gc

output_file = "saved_emb/ccm_1000_ave_0620.txt"
len_criteria = 1000

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D") # esm2_t6_8M_UR50D, esm2_t33_650M_UR50D, esm_msa1b_t12_100M_UR50S
layer_num = 33
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
df = pd.read_csv("train_data/trainingmodellarge_0620.csv")
df.rename(columns={"Uniprot/Description": "description", "Fasta Seq": "seq", "Identification (output)": "output"}, inplace=True)

#print(df.loc[1662])

#from left data
# Factorize output names to numbers
df['output'] = pd.factorize(df['output'])[0]
seq_list = df['seq'].tolist()
output_list = df['output'].tolist()

# truncate len(seq)>2000 and padding with "_"
seq = []
output = []
for i in range(0,len(seq_list)):
    m = len(seq_list[i])
    if m<=len_criteria:  #padding
        seq.append(seq_list[i])
        output.append(output_list[i])
    else: #truncate
        seq.append(seq_list[i][0:len_criteria])
        output.append(output_list[i])

data = list(zip(output, seq))
#data = data[0:2]
#print(data)



"""
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein1", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGILAGG"),
]
"""
batch_size = 2
num_samples = len(output_list)
sequence_representations_np_sum = []
batch_labels_np_sum = []
for n in range(0, num_samples, batch_size):
    batch_data = data[n : n + batch_size]
    if len(batch_data)!=2:
        break

    # labels, sequence strs, tokens that transfer sequence strs to numbers
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  #sequence length
    #print(batch_labels)
    #print(batch_tokens)
    #print(batch_lens)

    # Extract per-residue representations (on CPU)
    #print("1111111111111111111111")
    #print(len(np.array(batch_tokens)[1]))  #1002

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer_num], return_contacts=True)
    #print("2222222222222222222222")
    #token_representations = results["representations"][33]
    # Move the token_representations to the CPU
    token_representations = results["representations"][layer_num]


    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    #print(sequence_representations)

    # Convert sequence representations and batch labels to numpy arrays
    #sequence_representations_np = np.array(sequence_representations)
    #batch_labels_np = np.array(batch_labels)
    # Convert the list of tensors to a stacked tensor
    stacked_tensor_seq = torch.stack(sequence_representations)
    # Convert the stacked tensor to a NumPy array
    sequence_representations_np = stacked_tensor_seq.numpy()
    # Convert the list of tensors to a NumPy array
    #sequence_representations_np = np.array([tensor.numpy() for tensor in stacked_tensor_seq])

    #print(len(sequence_representations_np[0]))
    #print(len(sequence_representations_np))
    #print(len(sequence_representations_np[1])) #320

    #sequence_representations_np_sum.append(sequence_representations_np)
    #batch_labels_np_sum.append(batch_labels_np)
    # Reshape the array to 1-dimensional

    #batch_df = pd.DataFrame({"seq": sequence_representations_np}, index=range(len(sequence_representations_np)))
    # Append the batch DataFrame to the CSV file
    #batch_df.to_csv(output_file, mode='a', header=not os.path.isfile(output_file), index=False)
    with open(output_file, 'a+') as file:
        # Save the array to the file using numpy.savetxt
        #np.savetxt(file, sequence_representations_np[0])
        # Optionally, you can add a delimiter between each array
        np.savetxt(file, [sequence_representations_np[0]], delimiter=',', newline='', fmt='%.8f')
        file.write('\n')
        np.savetxt(file, [sequence_representations_np[1]], delimiter=',', newline='', fmt='%.8f')
        # Optionally, you can add a delimiter between each array
        file.write('\n')

    # Clear CPU memory
    #del batch_data
    #del batch_df
    #del sequence_representations_np
    #del batch_labels_np
    #del batch_labels, batch_strs, batch_tokens, batch_lens, results, token_representations, sequence_representations
    # Clear GPU memory
    #torch.cuda.empty_cache()

    print(n)

# Perform final garbage collection
#gc.collect()
# Create a DataFrame with the sequence representations and batch labels
#df = pd.DataFrame({"Seq": sequence_representations_np_sum, "output": batch_labels_np_sum})
# Save the DataFrame to a CSV file
#df.to_csv("saved_emb/esm2_t33_650M_UR50D.csv", index=False)

"""
# Look at the unsupervised self-attention map contact predictions
import matplotlib.pyplot as plt
for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    plt.title(seq)
    plt.show()
"""


"""
batch_labels
sequence_representations


batch_size = 128
epochs = 100
k_num=0
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
frac = 0.5

for train, test in kfold.split(X, Y):
    k_num+=1
    x_train = X[train]
    x_test = X[test]
    y_train = Y[train]
    y_test = Y[test]

    # Define the paths to save the best models
    checkpoint_path = 'best_model' + str(k_num) +'.h5'
    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_precision', mode='max', patience=10, restore_best_weights=True)
    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_precision', save_best_only=True)

    model = create_model(len_criteria, 1)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  # 'sgd',tf.keras.optimizers.Adam(learning_rate=0.001),  #
                  loss=loss_function, #'binary_crossentropy',
                  # loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=METRICS)

    history = model.fit(x_train, y_train, #x_gen,
                        epochs=epochs,
                        verbose=1, #show epoch training process 0, 1, 2
                        validation_data=(x_test, y_test),
                        #callbacks=[ohem_callback],
                        #callbacks=[early_stopping, checkpoint],
                        batch_size=batch_size)  # class_weight=weights_dict

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

print("training.......................\n")
print(training_f1_kfold)
print(training_specificity_kfold)
print(training_sensitivity_kfold)
print(training_accuracy_kfold)

print("training average.......................\n")
print(Average(training_f1_kfold))
print(Average(training_specificity_kfold))
print(Average(training_sensitivity_kfold))
print(Average(training_accuracy_kfold))

print("testing.......................\n")
print(validation_f1_kfold)
print(validation_specificity_kfold)
print(validation_sensitivity_kfold)
print(validation_accuracy_kfold)

print("testing average.......................\n")
print(Average(validation_f1_kfold))
print(Average(validation_specificity_kfold))
print(Average(validation_sensitivity_kfold))
print(Average(validation_accuracy_kfold))

print("training sample average.......................\n")
print(Average(training_NTF))
print(Average(training_TF))

print("testing sample average.......................\n")
print(Average(validation_NTF))
print(Average(validation_TF))

"""



