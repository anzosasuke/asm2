# import sys
# sys.path.append('home/ashish/Onedrive/research/mixed/Programs/asm2vec/')
# print(sys.path)
from asm2vec.model import Asm2Vec
from prepare_asm2vec import *
import argparse
import os
from tqdm import tqdm
import gc
from statistics import mean
from sklearn.model_selection import train_test_split


def expriment(d_exp, alpha, train, val_test, train_flags , val_test_flags):
    funcs_per_cwe = {}
    print("training starts for d=", d_exp,"and alpha=", alpha)
    model = Asm2Vec(neg_samples=25, d=d_exp, rnd_walks=10, initial_alpha=alpha)
    train_repo = model.make_function_repo(train)
    model.train(train_repo)
    print("training complete")
    ###########################
    print("estimating starts")
    estimating_funcs_vec = list(map(lambda f: model.to_vec(f), val_test))
    for (ef, efv) in zip(val_test, estimating_funcs_vec):
        estimate_index = val_test.index(ef)
        flag_val_test = val_test_flags[estimate_index]
        func_repo = []
        for tf in train_repo.funcs():
            index = train_repo.funcs().index(tf)
            sim = cosine_similarity(tf.v, efv)
            func_repo.append((tf.sequential().name(), train_flags[index], sim))
        func_repo.sort(key=lambda x: x[-1], reverse=True)
        acc = compute_acc_per_good_bad(flag_val_test, func_repo, 15)
        funcs_per_cwe[(flag_val_test, ef.name())] = acc
        print(acc)
    avg_funcs = compute_avg_per_cwe(funcs_per_cwe)
    print('Estimating complete.')
    return avg_funcs




def find_best_d_alpha(averages):
    averages.sort(key=lambda x: x[2]+x[3], reverse=True)

    return averages[0]

# Making changes to fit any binary
if __name__ == '__main__':
    folder_path = os.getcwd()
    functs = []
    # print(folder_path)
    for root, dirs, files in os.walk(f'{folder_path}/bin'):
        # print(files)
        for file in files:
            full_path = os.path.join(root,file)
            # print(full_path)
            results = function_asm(full_path)
            for result in results:
                if result is not None:
                    cfg, entry_node, func_name = result
                    if cfg is not None and entry_node is not None:
                        flag, function = generate_asm2vec_function(result)
                        # print(flag, function)

                        if flag is True:
                            functs.append(function)

for fun in functs:
    print(fun)
    cvec = Asm2Vec()
    cvec.to_vec(fun)
    print(cvec)