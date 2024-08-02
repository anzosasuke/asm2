import concurrent.futures
import numpy as np


import sys
sys.setrecursionlimit(10000)
from asm2vec.model import Asm2Vec
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import dill
import gc

def deserialize_from_pickle(filename):
    with open(filename, 'rb') as f:
        return dill.load(f)
def process_chunk(funcs_chunk, loaded_model):
    funcs = set()
    chunk_vectors = []

    for func in funcs_chunk:
        if func.name().startswith("_") or func.name() in funcs:
            continue
        chunk_vectors.append(loaded_model.to_vec(func))
        funcs.add(func.name())
    print(f"Processed {len(chunk_vectors)} functions in chunk")
    return chunk_vectors

def chunkify(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def write_numpy_vector1(loaded_model, functions_list_pickle, output_path, total_vectors=6000, batch_size=100):
    vect_ = []
    functions_list = deserialize_from_pickle(functions_list_pickle)  # Deserialize the list of functions
    functions_list = functions_list[:total_vectors]
    chunks = chunkify(functions_list, batch_size)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_chunk, chunk, loaded_model): chunk for chunk in chunks}

        for future in concurrent.futures.as_completed(futures):
            chunk_vectors = future.result()
            vect_.extend(chunk_vectors)
            print("Total collected vectors so far:", len(vect_))
            if len(vect_) >= total_vectors:
                vect_ = vect_[:total_vectors]
                break

    vect_ = np.array(vect_)
    np.save(output_path, vect_)

    # Explicitly delete large objects and call garbage collection
    del vect_
    del functions_list
    gc.collect()


def process_functions(functions, loaded_model):
    funcs = set()
    file_vectors = []

    for func in functions:
        if func.name().startswith("_") or func.name() in funcs:
            continue
        file_vectors.append(loaded_model.to_vec(func))
        funcs.add(func.name())
    print(len(funcs))
    return file_vectors
def write_numpy_vector(loaded_model, functions_list_pickle, output_path, batch_size=500):
    vect_ = []
    functions_list = deserialize_from_pickle(functions_list_pickle)  # Deserialize the list of functions

    file_vectors = process_functions(functions_list, loaded_model)
    vect_.extend(file_vectors)
    print(len(vect_))

    vect_ = np.array(vect_)
    np.save(output_path, vect_)

    # Explicitly delete large objects and call garbage collection
    del vect_
    del functions_list
    gc.collect()

functions_list_pickle = 'function_pickle/optimized/total/c.pkl'
output_path = 'vectors/optimized/c/c.npy'
loaded_model = deserialize_from_pickle('models/optimized/first2k_c10.pkl')
print("load1")
write_numpy_vector(loaded_model, functions_list_pickle, output_path)

functions_list_pickle = 'function_pickle/optimized/total/cpp.pkl'
output_path = 'vectors/optimized/cpp/cpp.npy'
loaded_model = deserialize_from_pickle('models/optimized/first2k_cpp10.pkl')
print("load2")
write_numpy_vector(loaded_model, functions_list_pickle, output_path)

functions_list_pickle = 'function_pickle/optimized/total/go.pkl'
output_path = 'vectors/optimized/go/go.npy'
loaded_model = deserialize_from_pickle('models/optimized/first2k_rust10.pkl')
print("load3")
write_numpy_vector(loaded_model, functions_list_pickle, output_path)

functions_list_pickle = 'function_pickle/optimized/total/rust.pkl'
output_path = 'vectors/optimized/rust/rust.npy'
loaded_model = deserialize_from_pickle('models/optimized/first2k_go10.pkl')
print("load4")
write_numpy_vector(loaded_model, functions_list_pickle, output_path)


for i in range(3):
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))