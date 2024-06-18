from asm2vec.parse import parse_fp
from asm2vec.model  import Asm2Vec

with open('ac1.asm', 'r') as fp:
    funcs = parse_fp(fp)
    print(len(funcs))
