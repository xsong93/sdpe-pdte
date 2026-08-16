//
// Created by Xintong Song on 2026/5/22.
//


#ifndef V2I_DECISION_TREE_H
#define V2I_DECISION_TREE_H

struct Node {
    bool isLeaf{false};
    int featureIdx{-1};
    int threshold{0};
    int parentIdx{-1};
    int leftChild{-1};
    int rightChild{-1};
    int depthOnTree{0};
    int leafValueLeft{0};
    int leafValueRight{0};
    int pathValueLeft{0};
    int pathValueRight{0};
};

#endif // V2I_DECISION_TREE_H
