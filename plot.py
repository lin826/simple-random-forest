from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus

def plot_decision_tree(l_tree,f):
    tree.export_graphviz(l_tree[0], out_file=f+"_1",
                         filled=True, rounded=True,
                         special_characters=True)

    tree.export_graphviz(l_tree[1], out_file=f+"_2",
                         filled=True, rounded=True,
                         special_characters=True)

    tree.export_graphviz(l_tree[2], out_file=f+"_3",
                         filled=True, rounded=True,
                         special_characters=True)
    pass
