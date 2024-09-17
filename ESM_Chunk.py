import os
import numpy as np
import torch
import glob
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

path = 'trainingmodellarge_0620_full_L6/*.*'
files = glob.glob(path)
X = []
X_chunk = []
print("appending data...")
for name in files:
    with open(name) as f:
     embedding = torch.load(name)
     embedding = embedding['representations'][6].numpy()
     if len(embedding) < 1022:
         padding = np.zeros(((1022-len(embedding)), 320))
         embedding = embedding.tolist() + padding.tolist()
         embedding = np.array(embedding)
     X.append(embedding)

print("averaging...")
counter = 0
i = 0
avg = []
temp = np.zeros(320)
for sample in X:
    for emb in sample:
        temp = np.add(temp, emb)
        if counter == 100:
            avg.append(temp)
            temp = np.zeros(320)
            counter = 0
        counter+=1
    counter = 0
    X_chunk.append(avg)
    temp = np.zeros(320)
    avg = []
    i+=1

#print(np.shape(X_chunk))
np.save("X_chunk", X_chunk)
