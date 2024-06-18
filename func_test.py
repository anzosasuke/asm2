from asm2vec.model import Asm2Vec
from asm2vec.repo import *
import pickle
import json
from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import parse_instruction


with open('asm2vec_func.pkl', 'rb') as f:
    asm2vec_func = pickle.load(f)


    print(asm2vec_func)