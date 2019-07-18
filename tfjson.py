import json
import os, sys
import numpy as np

def findlist(json_obj,target):
    result=[[] for i in range(len(target))]

    def json_walk(node_name, path, node, targets):
        ty_node = type(node)
        #print(path,node_name,ty_node)
        if ty_node is dict: # Has child
            for k in node.keys():
                json_walk(k, path+'/'+k, node[k], targets)
        elif ty_node is list:
            if path in targets:
                rindx = targets.index(path)
                result[rindx].append(node)
            else:
                for n in node:
                    json_walk(node_name, path, n, targets)

    json_walk('','',json_obj,target)
    return result

def walk_tensor(json_obj,start_idx,generators):
    for j in json_obj:
        for o_idx in j['outputs']:
            generators[o_idx] = j['inputs']

if __name__=='__main__':
    with open('detect.json') as r:
        target=['/buffers/data','/operator_codes','/subgraphs/inputs', '/subgraphs/outputs']
        j = json.load(r)
        inputs    = findlist(j,['/subgraphs/inputs'])
        outputs   = findlist(j,['/subgraphs/outputs'])
        tensors   = findlist(j,['/subgraphs/tensors'])
        operators = findlist(j,['/subgraphs/operators'])
        buffers   = findlist(j,['/buffers'])
        print("inputs:",inputs[0][0])
        print("outputs:",outputs[0][0])
        print("tensors   N:",len(tensors[0][0]))
        print("buffers   N:",len(buffers[0][0]))
        print("operators N:",len(operators[0][0]))

        for idx, start_operator in enumerate(operators[0][0]):
            start_tensors = start_operator.get('inputs')
            if inputs[0][0][0] in start_tensors:
                break

        for t_idx in start_tensors:
            print(t_idx)
            tensor = tensors[0][0][t_idx]
            bufidx = tensor.get('buffer')
            bufshp = tensor.get('shape')
            print('tensor',tensor, bufidx, bufshp, np.prod(bufshp))
            bufdata= buffers[0][0][bufidx]
            if bufdata=={}:
                bN = 0
            else:
                bN = len(bufdata.get('data'))
            print('buffer',bufidx,bufdata,bN)

        #list_generators = [[-1] for i in range(len(tensors[0][0]))]
        #walk_tensor(operators[0][0], inputs[0][0][0], generators)
        #for i in range(len(generators)):
            #print(i,generators[i])

