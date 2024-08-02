import sys
sys.setrecursionlimit(10000)
from asm2vec.model import Asm2Vec
from asm2vec.repo import *
import pickle
import argparse
import matplotlib.pyplot as plt
import os
import dill
from scipy.spatial.distance import minkowski
def deserialize_from_pickle(filename):
    with open(filename, 'rb') as f:
        loaded_model = dill.load(f)
        # functiion_vec = {func.name: model.to_vec(func) for func in train_repo}
        return loaded_model


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def compare_2_functions(asm2vec_fun1, asm2vec_fun2, model, model1):
            vec1 = model.to_vec(asm2vec_fun1)
            vec2 = model1.to_vec(asm2vec_fun2)

            # return cosine_similarity(vec1, vec2)
            return cosine_similarity(vec1, vec2)

# file = os.path.join("functions/c/", "spec_compress")
# file2 = os.path.join("functions/rust/", "reverse")
# model = os.path.join("models", "first2k_rust5.pkl")
# func1 = deserialize_from_pickle(file)
# func2 = deserialize_from_pickle(file2)
# model = deserialize_from_pickle(model)
# print(compare_2_functions(func1, func1, model))
# print(compare_2_functions(func2, func2, model))
# print(compare_2_functions(func2, func1, model))

def process_file(c_func, loaded_model, loaded_model1, rust_func):
    """Helper function to compare a C function to the Rust function."""
    # c_func = deserialize_from_pickle(c_file)

    # if rust_func != None:
    #     for i, file in enumerate(os.listdir(rust_func)):
    #         # if i >= 10:
    #         #     break
    #         r_func = deserialize_from_pickle(rust_func+ str(file))
    #         # rust_func = deserialize_from_pickle(os.path.join('functions/Rust', rust_file))
    #         sim = compare_2_functions(c_func, r_func, loaded_model)
    #
    #         print(c_func.name()[:15], sim, r_func.name()[:15])
    #         return sim

    if rust_func:
        for i, file in enumerate(deserialize_from_pickle(rust_func)):
            if i > 10:
                sim = compare_2_functions(c_func, file, loaded_model, loaded_model1)
                print(c_func.name()[:15], sim, file.name()[:15])
                return sim


    else:
        # c_func = deserialize_vocabulary(c_file)
        # rust_func = deserialize_from_pickle(os.path.join('functions/Rust', rust_file))
        sim = compare_2_functions(c_func, c_func, loaded_model, loaded_model1)

        print(c_func.name()[:15], sim)
        return sim


# vec = np.load('renew_total.npy', allow_pickle=True)
# print(vec[:5])

if __name__ == '__main__':
    model = deserialize_from_pickle('models/optimized/first2k_go10.pkl')
    model1 = deserialize_from_pickle('models/optimized/first2k_cpp10.pkl')
    folder = deserialize_from_pickle('function_pickle/optimized/total/c.pkl')
    for i, file in enumerate(folder):
        if i >= 40:
            break

        # if str(file) == "first100":
        # compare = process_file(c_file='functions/rust/'+str(file), loaded_model=model, rust_func='functions/c/')
        compare = process_file(c_func=file, loaded_model=model1, loaded_model1=model1, rust_func=None)


