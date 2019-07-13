import json
import os, sys

def nodep(node_name, node):
    ty = type(node)
    if ty is dict: # Has child
        for k in node.keys():
            nodep(k, node[k])
    elif ty is list:
        if node_name == 'data':
            print(len(node))
        else:
            for n in node:
                nodep(node_name, n)

with open('detect.json') as r:
    j = json.load(r)
    nodep('/',j)
