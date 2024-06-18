import numpy as np
import sys
sys.setrecursionlimit(10000)
from asm2vec.model import Asm2Vec
from asm2vec.repo import *
import pickle
import json
from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import parse_instruction
import os
import argparse
import matplotlib.pyplot as plt
def write_function_pkl(asm2vec_functions):
    for asm2vec_function in asm2vec_functions:
        name_ = asm2vec_function.name()
        if len(name_) > 30:
            name_ = name_[:30]
            folder = os.path.join('functions/c/', name_)
            with open(folder, 'wb') as f:
               pickle.dump(asm2vec_function, f)

        else:
            folder = os.path.join('functions/c/', name_)
            with open(folder, 'wb') as f:
                pickle.dump(asm2vec_function, f)

def deserialize_from_pickle(filename):
    with open(filename, 'rb') as f:
        loaded_model = pickle.load(f)
        # functiion_vec = {func.name: model.to_vec(func) for func in train_repo}
        return loaded_model
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def compare_2_functions(pkl1, pkl2, model):
            vec1 = model.to_vec(pkl1)
            vec2 = model.to_vec(pkl2)
            return cosine_similarity(vec1, vec2)

def write_vector(model):

    for file in os.listdir('functions/c/'):
        file1 = os.path.join('functions/c/', file)
        file1 = deserialize_from_pickle(file1)
        folder = os.path.join('vectors/C/', str(file))
        vector = model.to_vec(file1)
        with open(folder, 'wb') as f:
           pickle.dump(vector, f)


def fetch_functions():
    parser = argparse.ArgumentParser(description= "Read two files")
    parser.add_argument('file1', type=argparse.FileType('rb'), help='First file')
    parser.add_argument('file2', type=argparse.FileType('rb'), help='second file')
    args = parser.parse_args()

    pkl1 = pickle.load(args.file1)

    pkl2 = pickle.load(args.file2)
    return pkl1, pkl2



loaded_model = deserialize_from_pickle('asm2vec_model1.pkl')



# write_vector(loaded_model) # taking the individual vector out of the function pickle



#################### Use when you want to get the individual functions from big pickle ########
# asm2vec_funcs = deserialize_from_pickle('function_pickle/C/sgcc')
# write_function_pkl(asm2vec_funcs)
vect_ = []
def write_numpy_vector(loaded_model):
    for func in asm2vec_funcs:
        vect_.append(loaded_model.to_vec(func))
    # print(np.array(vect_).shape)
    folder = os.path.join('vectors/c_big', "c_vec" )
    vect_ = np.array(vect_)
    np.save(folder, vect_)

# print(vect_[:5])
################################################################

###################### Check for Smiliarity ###################
# pkl1, pkl2 = fetch_functions()
# sim = compare_2_functions(pkl1, pkl2, loaded_model)
# print(sim)
##############################################################

############### New Experiment #############

def vector_append(asm2vec_funcs, model):
    vectors = []

    for funcs in asm2vec_funcs:
        vector = model.to_vec(funcs)
        if len(vectors) > 0:
            vectors = np.insert(vectors, vector.shape, vector, axis=0)
        else:
            vectors = np.append(vectors, vector, axis=0)
    return vectors
# vec1 = loaded_model.to_vec(pkl1)
# print(type(vec1))
# print(vec1.ndim)
# print(vec1.shape)
# print(vec1.size)



def vector_print(asm2vec_funcs, model):
    for funcs in asm2vec_funcs:
        vector = model.to_vec(funcs)
        plt.plot(vector)
        plt.title(funcs.name())
        plt.show()


# vectors = vector_append(asm2vec_funcs, loaded_model)
# print(vectors.ndim)
# print(vectors.shape)
# print(vectors.size)
# plt.plot(vectors)
# plt.show()


###### vector plot in line graph #################
# vector_print(asm2vec_funcs, loaded_model)
##############################

# print(len(asm2vec_funcs))


folder = os.path.join('vectors/Rust', "rust_vec.npy" )
vect_ = np.load(folder)
print(vect_.shape)
print(vect_[0].size)
