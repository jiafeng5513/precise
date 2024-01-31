import numpy as np
import wavio
import os
from precise import vectorization

dataset_path = "/home/anna/WorkSpace/celadon/demo-src/precise/training/data/wake-word"

for filename in os.listdir(dataset_path):
    wav_path = os.path.join(dataset_path, filename)
    wav = wavio.read(wav_path)
    print(filename)
    print("wav.data.shape = {}".format(wav.data.shape))
    data = np.squeeze(wav.data)
    print("data.shape = {}".format(data.shape))
    result = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    print("result.shape = {}".format(result.shape))
    vec = vectorization.vectorize(result)
    print("vec.shape = {}".format(vec.shape))