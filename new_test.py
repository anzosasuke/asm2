from asm2vec.parse import parse_fp
from asm2vec.model  import Asm2Vec

with open('disasm.asm', 'rb') as fp:
    funcs = parse_fp(fp)
    print(len(funcs))
