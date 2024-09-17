import os
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X = np.load("X_S_IG.npy")
Y = np.load("Y_S.npy")

"""
model = tf.keras.models.load_model('saved_model/CNN/model_0622_1')

baseline_input = tf.zeros(shape=(1,))

attributions_list = []
counter = 0

def integrated_gradients(inputs):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        scaled_inputs = [baseline_tensor + (inputs - baseline_tensor) * alpha for alpha in np.linspace(0, 1, num=100)]
        outputs = [model(tf.expand_dims(inputs, axis=0)) for inputs in scaled_inputs]
    grads = [tape.gradient(output, inputs) for output in outputs]
    avg_grads = tf.reduce_mean(grads, axis=0)
    attributions = (inputs - baseline_tensor) * avg_grads
    return attributions


for preprocessed_input in X:
    input_tensor = tf.convert_to_tensor(preprocessed_input, dtype=tf.float32)
    baseline_tensor = tf.convert_to_tensor(baseline_input, dtype=tf.float32)
    attributions = integrated_gradients(input_tensor)
    attributions_list.append(attributions)

    counter += 1
    print(counter, " of ", len(X))

np.save("X_S_IG.npy", attributions_list)
"""
TF_atr = []
NTF_atr = []
for i in range(len(X)):
    if Y[i] == 1: TF_atr.append(X[i])
    else: NTF_atr.append(X[i])

TF_avg = np.mean(TF_atr, axis=0)
NTF_avg = np.mean(NTF_atr, axis=0)
diff = TF_avg - NTF_avg

plt.figure(figsize=(10, 4))
plt.bar(range(len(diff)), diff)
plt.xlabel('Feature Index')
plt.ylabel('Attribution')
plt.title('Integrated Gradients - TF Feature Attributions')
plt.show()
