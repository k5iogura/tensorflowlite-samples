import json
import os, sys

target=['/buffers/data','/operator_codes','/subgraphs/inputs', '/subgraphs/outputs']
result=[[] for i in range(len(target))]

def findlist(node_name, path, node, targets):
    ty_node = type(node)
    #print(path,node_name,ty_node)
    if ty_node is dict: # Has child
        for k in node.keys():
            findlist(k, path+'/'+k, node[k], targets)
    elif ty_node is list:
        if path in targets:
#            print(path,node)
            rindx = targets.index(path)
            result[rindx].append(node)
        else:
            for n in node:
                findlist(node_name, path, n, targets)

with open('detect.json') as r:
    j = json.load(r)
    findlist('','',j,target)
    for i,r in enumerate(result):
        if len(r)>100:
            for j in r:
                print(target[i],len(j))
        else:
            print(target[i],r)

