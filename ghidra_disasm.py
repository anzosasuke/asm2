from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import parse_instruction
from ghidra.program.model.block import BasicBlockModel
from ghidra.program.model.block import *
from ghidra.util.task import ConsoleTaskMonitor
from asm2vec.model import Asm2Vec
from asm2vec.repo import *
import pickle
import json
import os
import numpy as np
import sys
sys.setrecursionlimit(20000)
import dill
def ghidra_to_asm2vec(functions):
    asm2vec_functions = []
    function_map = {}
    for function in functions:
        addrSet = function.getBody()
        codeUnits1 = listing.getCodeUnits(addrSet, True)
        if len(list(codeUnits1)) > 5:
            asm2vec_blocks = []
            block_map = {}

            codeBlockIterator = bbm.getCodeBlocksContaining(function.getBody(), monitor)
            while codeBlockIterator.hasNext():
                block = codeBlockIterator.next()
                asm2vec_block = BasicBlock()

                codeUnitIterator = listing.getCodeUnits(block, True)
                while codeUnitIterator.hasNext():
                    codeUnit = codeUnitIterator.next()
                    instruction = codeUnit.toString()
                    asm2vec_block.add_instruction(parse_instruction(instruction))

                block_map[block] = asm2vec_block
                asm2vec_blocks.append(asm2vec_block)

            for block, asm2vec_block in block_map.items():
                destIterator = block.getDestinations(monitor)
                while destIterator.hasNext():
                    destination = destIterator.next()
                    des_block = destination.getDestinationBlock()
                    if des_block in block_map:
                        asm2vec_block.add_successor(block_map[des_block])
                        block_map[des_block].add_predecessor(asm2vec_block)

            asm2vec_function = Function(asm2vec_blocks[0], function.getName())
            asm2vec_functions.append(asm2vec_function)
            function_map[function] = asm2vec_function

    for function in functions:
        asm2vec_function = function_map.get(function)
        if asm2vec_function is None:
            continue

        codeUnits2 = listing.getCodeUnits(function.getBody(), True)
        for codeUnit in codeUnits2:
            if codeUnit.getFlowType().isCall():
                callee = codeUnit.getDefaultFlows()
                callee = [call for call in callee]
                if len(callee) == 1:
                    callee_f = listing.getFunctionAt(callee[0])
                    callee_function = function_map.get(callee_f)
                    if callee_function is not None:
                        # print("here")
                        asm2vec_function.add_callee(callee_function)
                        asm2vec_function.add_caller(asm2vec_function)


    return asm2vec_functions



bbm = BasicBlockModel(currentProgram())
listing = currentProgram().getListing()
monitor = ConsoleTaskMonitor()
functions = currentProgram().getFunctionManager().getFunctions(True)
asm2vec_func = ghidra_to_asm2vec(list(functions))
# print(asm2vec_func)
# print(len(asm2vec_func))
# for fun in asm2vec_func:
#     print(fun._name)
#
name = str(currentProgram().getExecutablePath()).split('/')[-1]
folder_path = os.path.join('function_pickle/optimized/rust/', name.split('.')[0])
asm2vec_func = asm2vec_func[:10000]
with open(folder_path, 'wb') as f:
    dill.dump(asm2vec_func, f)

# model = Asm2Vec(d=200)
# train_repo = model.make_function_repo(asm2vec_func)
# model.train(train_repo)



#
#
# #
# with open('asm2vec_model1.pkl', 'wb') as f:
#     pickle.dump(model, f)




# serialize_to_json(asm2vec_functions, 'asm2vec_model.json')
