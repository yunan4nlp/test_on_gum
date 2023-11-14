import sys
sys.path.extend(["../../","../","./"])
import random
import argparse
import pickle
import time

from data.Config import *
from data.Dataloader import *
from modules.Parser import *
from modules.EDULSTM import *
from modules.Decoder import *
from modules.XLNetTune import *
from data.TokenHelper import *
from modules.GlobalEncoder import *
from modules.Optimizer import *
from data.GumDataLoader import *
from collections import Counter

f2cMap = {
    "joint": "list",
    "elaboration": "elab",
    "sequence": "temp",
    "same-unit": "same",
    "contrast": "cont",
    "attribution": "attr",
    "preparation": "preparation",

    "background": "back",
    "circumstance": "back",

    "restatement": "summ",
    "evaluation": "eval",
    "evidence": "evid",
    "purpose": "purp",
    "concession": "cont",

    "cause": "cause",
    "result": "cause",

    "condition": "cond",
    "antithesis": "cond",

    "justify": "justify",

    "question": "prob",
    "manner": "mann",

    "motivation": "motivation",
    "means": "mann",
    "solutionhood": "prob",

    span_str:span_str,
    root_label:root_label
}

def create_relation_vocab(gum_data):
    outf = open("relation_list", mode='w', encoding="utf8")

    label_counter = Counter()
    for doc in gum_data:
        for node in doc._all_nodes:
            if node.label != span_str and node.label != root_label:
                label_counter[node.label] += 1
    
    for label, count in label_counter.most_common():
        outf.write(label + "\n")

    outf.close()
    return


def f2c(doc):
    for node in doc._all_nodes:
        assert node.label in f2cMap
        c_label = f2cMap[node.label]
        node.label = c_label
    return


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--f_dir', default='gum_rst_lisp_binary_f', help='without evaluation')
    argparser.add_argument('--c_dir', default='gum_rst_lisp_binary_c', help='without evaluation')

    args, extra_args = argparser.parse_known_args()

    for file_name in os.listdir(args.f_dir):
        if file_name[-3:] != "dis": continue
        gum_file_path = os.path.join(args.f_dir, file_name)
        print("file: ", gum_file_path)
        doc = read_gum_corpus_ll(gum_file_path)

        f2c(doc)

        out_file_path = os.path.join(args.c_dir, file_name + '.crelation')
        with open(out_file_path, mode='w', encoding='utf8') as out_f:
            tree = str(doc._root_node)
            out_f.write(tree)
    #create_relation_vocab(gum_data)