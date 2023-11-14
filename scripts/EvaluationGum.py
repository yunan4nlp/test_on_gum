import sys
sys.path.extend(["../../","../","./"])

from data.Config import *
from data.Dataloader import *
from data.GumDataLoader import *

import argparse

def labelled_attachments(doc):

    spans = set()
    nuclears = set()
    relations = set()
    fulls = set()

    assert len(doc._nonleaf_nodes) + 1 == len(doc._leaf_nodes)

    for node in doc._nonleaf_nodes:
        span = str(node.edu_start) + "#" + str(node.edu_end)
        spans.add(span)

        n_label = node.left_node.nuclearity + "#" + node.right_node.nuclearity
        nuclear = span + "#" +  n_label
        nuclears.add(nuclear)

        if node.left_node.nuclearity == n_str and node.right_node.nuclearity == n_str:
            r_label = node.left_node.label
        elif node.left_node.nuclearity == s_str and node.right_node.nuclearity == n_str:
            assert node.right_node.label == span_str
            r_label = node.left_node.label
        elif node.left_node.nuclearity == n_str and node.right_node.nuclearity == s_str:
            assert node.left_node.label == span_str
            r_label = node.right_node.label
        else:
            print("error node")
            exit()

        relation = span + "#" + r_label
        relations.add(relation)

        full = span + "#" + n_label + "#" + r_label
        fulls.add(full)
        
    return spans, nuclears, relations, fulls


def evaluation(predict_file, gold_file):


    predict_doc = read_gum_corpus_ll(predict_file)
    gold_doc = read_gum_corpus_ll(gold_file)

    pred_spans, pred_nuclears, pred_relations, pred_fulls = labelled_attachments(predict_doc)
    gold_spans, gold_nuclears, gold_relations, gold_fulls = labelled_attachments(gold_doc)

    s_metric =  Metric()
    s_metric.correct_label_count = len(pred_spans & gold_spans)
    s_metric.predicated_label_count = len(pred_spans)
    s_metric.overall_label_count = len(gold_spans)

    n_metric = Metric()
    n_metric.correct_label_count = len(pred_nuclears & gold_nuclears)
    n_metric.predicated_label_count = len(pred_nuclears)
    n_metric.overall_label_count = len(gold_nuclears)

    r_metric = Metric()
    r_metric.correct_label_count = len(pred_relations & gold_relations)
    r_metric.predicated_label_count = len(pred_relations)
    r_metric.overall_label_count = len(gold_relations)

    f_metric = Metric()
    f_metric.correct_label_count = len(pred_fulls & gold_fulls)
    f_metric.predicated_label_count = len(pred_fulls)
    f_metric.overall_label_count = len(gold_fulls)
    return s_metric, n_metric, r_metric, f_metric

def test_dir(path):

    overall_s = Metric()
    overall_n = Metric()
    overall_r = Metric()
    overall_f = Metric()

    for file_name in os.listdir(path):
        if file_name[-9:] != "crelation": continue
        gold_file_path = os.path.join(path, file_name)
        print("file: ", gold_file_path)
        predict_file_path = gold_file_path + '.out'

        s_metric, n_metric, r_metric, f_metric = evaluation(predict_file_path, gold_file_path)

        overall_s.predicated_label_count += s_metric.predicated_label_count
        overall_s.correct_label_count += s_metric.correct_label_count
        overall_s.overall_label_count += s_metric.overall_label_count

        overall_n.predicated_label_count += n_metric.predicated_label_count
        overall_n.correct_label_count += n_metric.correct_label_count
        overall_n.overall_label_count += n_metric.overall_label_count

        overall_r.predicated_label_count += r_metric.predicated_label_count
        overall_r.correct_label_count += r_metric.correct_label_count
        overall_r.overall_label_count += r_metric.overall_label_count

        overall_f.predicated_label_count += f_metric.predicated_label_count
        overall_f.correct_label_count += f_metric.correct_label_count
        overall_f.overall_label_count += f_metric.overall_label_count

        s_metric.print()
        n_metric.print()
        r_metric.print()
        f_metric.print()

    print("overall: ") 
    overall_s.print()
    overall_n.print()
    overall_r.print()
    overall_f.print()

def test_domain_dir(path, domain):
    overall_s = Metric()
    overall_n = Metric()
    overall_r = Metric()
    overall_f = Metric()

    for file_name in os.listdir(path):
        if file_name[-9:] != "crelation": continue
        info = file_name.split("_")
        if info[1] != domain: continue

        gold_file_path = os.path.join(path, file_name)
        print("file: ", gold_file_path)
        predict_file_path = gold_file_path + '.out'

        s_metric, n_metric, r_metric, f_metric = evaluation(predict_file_path, gold_file_path)

        overall_s.predicated_label_count += s_metric.predicated_label_count
        overall_s.correct_label_count += s_metric.correct_label_count
        overall_s.overall_label_count += s_metric.overall_label_count

        overall_n.predicated_label_count += n_metric.predicated_label_count
        overall_n.correct_label_count += n_metric.correct_label_count
        overall_n.overall_label_count += n_metric.overall_label_count

        overall_r.predicated_label_count += r_metric.predicated_label_count
        overall_r.correct_label_count += r_metric.correct_label_count
        overall_r.overall_label_count += r_metric.overall_label_count

        overall_f.predicated_label_count += f_metric.predicated_label_count
        overall_f.correct_label_count += f_metric.correct_label_count
        overall_f.overall_label_count += f_metric.overall_label_count

        #s_metric.print()
        #n_metric.print()
        #r_metric.print()
        #f_metric.print()

    print("overall: ") 
    overall_s.print()
    overall_n.print()
    overall_r.print()
    overall_f.print()


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--test_dir', default='../baseline_predict/gum_rst_lisp_binary_c', help='')

    args, extra_args = argparser.parse_known_args()

    domain_list = ['academic', 'bio', 'conversation', 'fiction', 'interview', 'news', 'speech', 'textbook',  'vlog', 'voyage', 'whow']

    for domain_type in domain_list:
        test_domain_dir(args.test_dir, domain_type)
        print("domain:", domain_type)
        print('------')
