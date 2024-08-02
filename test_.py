import numpy as np
import sys
sys.setrecursionlimit(20000)
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
from multiprocessing import Pool
import dill
def write_in_pick(folder, name, model):
    with open(folder + '/' + name + '.pkl', 'wb') as f:
        dill.dump(model, f)

def write_function_pkl(asm2vec_functions, total_, funcs):
    for asm2vec_function in asm2vec_functions:
        name_ = asm2vec_function.name()
        if asm2vec_function.name().startswith("_"):
            continue
        for fun in funcs:
            if name_ == fun.name():
                if fun.__len__() == asm2vec_function.__len__():
                    continue

        funcs.add(asm2vec_function)
        total_.append(asm2vec_function)




# folder = os.path.join('function_pickle/opt/c/')
# total_c = []
# funcs = set()
# for file in os.listdir(folder):
#     with open(folder+file, "rb") as f:
#         write_function_pkl(pickle.load(f), total_c, funcs)
#
#     if len(total_c) > 6000:
#         total_c = total_c[:6000]
#         break
# # print("here")
# print(len(total_c))
#
# folder1 = os.path.join('function_pickle/optimized/total/')
# write_in_pick(folder1, "c", total_c)

# folder = os.path.join('function_pickle/optimized/cpp/')
# #
#
# total_cpp = []
# funcs = set()
#
# count = 0
# for file in os.listdir(folder):
#     try:
#         with open(folder+file, "rb") as f:
#             write_function_pkl(dill.load(f), total_cpp, funcs)
#             count = count+1
#             # print(count)
#     except EOFError:
#         print(file)
#         print("Error: The pickle file is invalid or empty.")
#     if len(total_cpp) >6000:
#         total_cpp = total_cpp[:6000]
#         break

# folder1 = os.path.join('function_pickle/optimized/total/')
# print("here")
# print(len(total_cpp))
#
# write_in_pick(folder1, "cpp", total_cpp)

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
    # change the address of the pcikle
    for file in os.listdir('function_pickle/Rust/'):
        file1 = os.path.join('function_pickle/Rust/', file)
        file1 = deserialize_from_pickle(file1)
        folder = os.path.join('vectors/Rust/', str(file))
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






# write_vector(loaded_model) # taking the individual vector out of the function pickle



#################### Use when you want to get the individual functions from big pickle ########
# asm2vec_funcs = deserialize_from_pickle('function_pickle/C/sgcc')
# write_function_pkl(asm2vec_funcs)

def write_numpy_vector(loaded_model, folder1, folder2):
    vect_ = []

    for i, file in enumerate(os.listdir(folder1)):
        file = os.path.join(folder1, file)
        # print(file)
        file = deserialize_from_pickle(file)
        # print("here")
        funcs = set()

        for func in file:
            if func.name().startswith("_") or func.name() in funcs:
                continue
            vect_.append(loaded_model.to_vec(func))
            funcs.add(func.name())
        print(len(funcs))
        if len(funcs) > 4000:
            break
        # print(np.array(vect_).shape)
    folder = folder2
    vect_ = np.array(vect_)
    np.save(folder, vect_)

# folder1= 'function_pickle/c'
# folder2 = 'vectors/c'
# loaded_model = deserialize_from_pickle('models/first2k_rust10.pkl')
# write_numpy_vector(loaded_model,folder1, folder2)
# folder1= 'function_pickle/rust'
# folder2 = 'vectors/rust'
# loaded_model = deserialize_from_pickle('models/first2k_10.pkl')
# write_numpy_vector(loaded_model, folder1, folder2)
# # print(vect_[:5])
################################################################

###################### Check for Smiliarity ###################

# pkl1, pkl2 = fetch_functions()
def process_file(c_file, loaded_model):
    """Helper function to compare a C function to the Rust function."""
    c_func = deserialize_from_pickle(os.path.join('functions/Rust', c_file))
    # rust_func = deserialize_from_pickle(os.path.join('functions/Rust', rust_file))
    sim = compare_2_functions(c_func, c_func, loaded_model)

    print(c_func.name(), sim)
    return sim



# c_files = sorted(os.listdir('functions/Rust'))[:100]  # Get first 100
#
# for c_file in c_files:
#     process_file(str(c_file), loaded_model)
# with Pool(processes=num_processes) as pool:
#     results = pool.map(process_file, file_pairs)
########################################################################
def first2000(folder):
    # breakpoint()
    # folder = os.path.join('function_pickle/Rust/')
    c_first5000 = []
    for file in os.listdir(folder):
        # print(file)
        c_func_pick = deserialize_from_pickle(folder+file)

        funcs = set()
        for func in c_func_pick:
            if len(c_first5000) < 2000:
                if func.name().startswith("_") or func.name() in funcs:
                    continue
                funcs.add(func.name())
                c_first5000.append(func)

    with open(os.path.join(folder+'first2000'), 'wb') as f:
        pickle.dump(c_first5000, f)
    # with open(os.path.join(folder[:folder.rfind('/')], "first100"), "wb") as f:
    #     pickle.dump(c_first5000, f)


# first2000(folder='function_pickle/c/')
# first = deserialize_from_pickle('function_pickle/c/alias')
# print(first)






# for file, sim in results:
#     print(f"Similarity for {file}: {sim}")
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

def write_big_vec(loaded_model):
    pass


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


# folder = os.path.join('vectors/Rust', "rust_vec.npy" )
# vect_ = np.load(folder)
# print(vect_.shape)
# print(vect_[0].size)



import os
import numpy as np
from multiprocessing import Pool
import multiprocessing

def process_file(file):
    print("############# New file is here############################ \n")
    vect_ = []
    file1 = os.path.join('function_pickle/Rust/', file)
    file1 = deserialize_from_pickle(file1)
    for func in file1:
        funcs = set()
        if func.name().startswith("_") or func.name() in funcs:
            continue
        # print(func.name())
        funcs.add(func.name())
        
        vect_.append(loaded_model.to_vec(func))
    folder = os.path.join('vectors/Rust/', file)
    vect_ = np.array(vect_)
    np.save(folder, vect_)

def write_numpy_vector1(loaded_model):
    files = os.listdir('function_pickle/Rust/')
    with Pool(os.cpu_count()) as p:
        p.map(process_file, files)

# write_numpy_vector1(loaded_model)



def train_model(folder, rnd_walks):
    # train_repo = Repo('train_repo')
    total = np.load(folder, allow_pickle=True)
    total = total[:2000]
    
    model = Asm2Vec(d = 200, rnd_walks = rnd_walks)
    train_repo = model.make_function_repo(total)
    model.train(train_repo)
    return model
# #
folder = [os.path.join('function_pickle/optimized/total/go.pkl'), os.path.join('function_pickle/optimized/total/rust.pkl')]
model = train_model(folder[0],3)
write_in_pick("models/optimized/", "first2k_go3", model=model )
model = train_model(folder[1],3)
write_in_pick("models/optimized", "first2k_rust3", model=model)
model = train_model(folder[0],5)
write_in_pick("models/optimized", "first2k_go5", model=model )
model = train_model(folder[1],5)
write_in_pick("models/optimized", "first2k_rust5", model=model)
model = train_model(folder[0],10)
write_in_pick("models/optimized", "first2k_go10", model=model )
model = train_model(folder[1],10)
write_in_pick("models/optimized", "first2k_rust10", model=model)

# pool = multiprocessing.Pool()
# model = pool.apply(train_model, ())
# model = train_model()

for i in range(3):
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))