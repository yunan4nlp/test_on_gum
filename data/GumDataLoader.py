from transition.State import *
from data.Discourse import *

root_label = 'root'
root_str = 'Root'
n_str = 'Nucleus'
s_str = 'Satellite'
span_str = "span"
leaf_str = "leaf"
class BinaryRstTree:
    def __init__(self) -> None:
        self.gold_actions = []
        self.words = []

        self.EDUs = []
        self._root_node = None

        self._all_nodes = list()
        self._leaf_nodes = list()
        self._nonleaf_nodes = list()

class RstNode(object):
    def __init__(self) -> None:
        self.left_node = None
        self.right_node = None
        self.parent_node = None

        self.edu_start = -1
        self.edu_end = -1

        self.nuclearity = None
        self.label = None
    
    def __str__(self) -> str:
        dep = self.depth()
        dep_spaces = ""
        rst_str = ""

        for idx in range(dep):
            dep_spaces += '  '

        if isinstance(self, RstLeafNode):
            span_info = "(" + leaf_str + " " + str(self.edu_start) + ")"
        else:
            span_info = "(" + span_str + " " + str(self.edu_start) + " " + str(self.edu_end)  + ")"
        
        relation_info = "(rel2par " + self.label + ")" 

        if isinstance(self, RstLeafNode):
            text = " ".join(self.words)
            text_info = "(text _!" + text + "_!)"
        
        if isinstance(self, RstLeafNode):
            rst_str = dep_spaces + "( " + self.nuclearity + " " + span_info + " " + relation_info + " " + text_info  + " )\n"
        else:
            if self.nuclearity == root_str:
                rst_str = dep_spaces + "( " + self.nuclearity + " " + span_info + "\n"
            else:
                rst_str = dep_spaces + "( " + self.nuclearity + " " + span_info + " " + relation_info + "\n"
            
            rst_str += str(self.left_node)
            rst_str += str(self.right_node)
            rst_str += dep_spaces + ")\n"
        return rst_str
    
    def depth(self):
        dep = 0
        p_node = self.parent_node
        while p_node != None:
            p_node = p_node.parent_node
            dep += 1
        return  dep


class RstLeafNode(RstNode):
    def __init__(self) -> None:
        RstNode.__init__(self)
        self.type = "<S>"
        self.words = []
        self.start = -1
        self.end = -1

def read_gum_corpus_ll(file_path):
    ### 

    _check_edu_stack = list()
    _n_stack = list()
    _l_stack = list()

    doc = BinaryRstTree()
    node_stack = list()

    doc.EDUs = list()

    with open(file_path, 'r') as infile:
        edu_start, edu_end = 0, 0

        for line in infile.readlines():
            line = line.strip()

            info_list = line.split(" ")
            if len(info_list) > 1: 
                node_type = info_list[2][1:]
                if node_type == span_str:
                    s_index, e_index = spanIndex(info_list)
                    n_label = nuclearity(info_list)
                    if n_label != root_str:
                        l_label = spanLabel(info_list)
                    else:
                        l_label = root_label
                    _check_edu_stack.append([s_index, e_index])
                    _n_stack.append(n_label)
                    _l_stack.append(l_label)

                elif node_type == leaf_str: ### shift
                    leaf_node = RstLeafNode()

                    words = leafWords(info_list)
                    if len(doc.words) == 0: 
                        edu_start = 0
                    else:
                        edu_start = len(doc.words) + 1 
                    edu_end = edu_start + len(words) - 1  

                    l_index = leafIndex(info_list)
                    leaf_node.label = leafLabel(info_list)
                    leaf_node.edu_start = l_index
                    leaf_node.edu_end = l_index
                    leaf_node.start = edu_start
                    leaf_node.end = edu_end
                    node_stack.append(leaf_node)

                    leaf_node.words = words
                    leaf_node.nuclearity = nuclearity(info_list)
                    doc.EDUs.append(leaf_node)
                    doc._all_nodes.append(leaf_node)
                    doc._leaf_nodes.append(leaf_node)

                    doc.words += words
                    ac = Action(CODE.SHIFT, -1, -1, leaf_str)
                    doc.gold_actions.append(ac)
                else:
                    print("error")

            elif len(info_list) == 1 and info_list[0] == ')': ### reduce
                assert len(node_stack) >= 2

                l_node = node_stack[-2]
                r_node = node_stack[-1]

                c_node = RstNode()
                c_node.left_node = l_node
                c_node.right_node = r_node
                c_node.edu_start = l_node.edu_start
                c_node.edu_end = r_node.edu_end
                l_node.parent_node = c_node
                r_node.parent_node = c_node

                if l_node.nuclearity == n_str and r_node.nuclearity == s_str:
                    nuclear = NUCLEAR.NS
                    label = r_node.label
                elif l_node.nuclearity == s_str and r_node.nuclearity == n_str: 
                    nuclear = NUCLEAR.SN
                    label = l_node.label
                elif l_node.nuclearity == n_str and r_node.nuclearity == n_str: 
                    nuclear = NUCLEAR.NN
                    assert l_node.label == r_node.label
                    label = l_node.label

                ac = Action(CODE.REDUCE, nuclear, -1, label)
                doc.gold_actions.append(ac)

                c_node.nuclearity = _n_stack[-1]
                c_node.label = _l_stack[-1]

                check_combine_node = _check_edu_stack[-1]
                assert c_node.edu_start == check_combine_node[0]
                assert c_node.edu_end == check_combine_node[1]

                node_stack.pop()
                node_stack.pop()
                node_stack.append(c_node)
                doc._all_nodes.append(c_node)
                doc._nonleaf_nodes.append(c_node)

                _check_edu_stack.pop()
                _n_stack.pop()
                _l_stack.pop()

    ac = Action(CODE.POP_ROOT)
    doc.gold_actions.append(ac)
    assert len(doc.EDUs) * 2 == len(doc.gold_actions)
    assert len(node_stack) == 1

    doc._root_node = node_stack[-1]

    return doc

def spanIndex(info_list):
    start_edu_index = int(info_list[3])
    end_edu_index = int(info_list[4][:-1])
    return start_edu_index, end_edu_index

def leafIndex(info_list):
    edu_index = int(info_list[3][:-1])
    return edu_index

def nuclearity(info_list):
    return info_list[1]

def spanLabel(info_list):
    return info_list[6][:-1]

def leafWords(info_list):
    words = info_list[7:-1]
    text = " ".join(words)
    info = text.split("_!")
    format_words = info[1].split(" ")
    return format_words

def leafLabel(info_list):
    return  info_list[5][:-1]

def getPredictRstTree(doc, vocab, states, step):
    pred_tree = BinaryRstTree()
    node_stack = list()
    for idx, state in enumerate(states):
        if idx >= step: break
        if idx == 0: continue

        action = state._pre_action
        if action.is_shift():
            leaf_node = RstLeafNode()
        
            edu_index = state._pre_state._next_index 

            edu = doc[0].EDUs[edu_index]
            leaf_node.words = edu.words

            leaf_node.edu_start = edu_index + 1
            leaf_node.edu_end = edu_index + 1
            node_stack.append(leaf_node)

            pred_tree._all_nodes.append(leaf_node)
            pred_tree._leaf_nodes.append(leaf_node)
        elif action.is_reduce():
            assert len(node_stack) >= 2

            l_node = node_stack[-2]
            r_node = node_stack[-1]
            c_node = RstNode()
            c_node.left_node = l_node
            c_node.right_node = r_node
            c_node.edu_start = l_node.edu_start
            c_node.edu_end = r_node.edu_end
            l_node.parent_node = c_node
            r_node.parent_node = c_node

            if action.nuclear == NUCLEAR.NN:
                l_node.nuclearity = n_str
                r_node.nuclearity = n_str

                l_node.label = vocab.id2rel(action.label)
                r_node.label = vocab.id2rel(action.label)

            elif action.nuclear == NUCLEAR.NS:
                l_node.nuclearity = n_str
                r_node.nuclearity = s_str

                r_node.label = vocab.id2rel(action.label)
                l_node.label = span_str
            elif action.nuclear == NUCLEAR.SN:
                l_node.nuclearity = s_str
                r_node.nuclearity = n_str

                l_node.label = vocab.id2rel(action.label)
                r_node.label = span_str

            node_stack.pop()
            node_stack.pop()
            node_stack.append(c_node)
            pred_tree._all_nodes.append(c_node)
            pred_tree._nonleaf_nodes.append(c_node)
    assert len(node_stack) == 1

    node_stack[0].nuclearity = root_str
    node_stack[0].label = root_label
    pred_tree._root_node = node_stack[0]

    return pred_tree

