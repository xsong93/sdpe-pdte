//
// Created by xsong93 on 6/15/26.
//

#ifndef THETAEQ0_DECISION_TREE_H
#define THETAEQ0_DECISION_TREE_H

#include <cstddef>
#include <vector>

#include "include/v2i_decision_tree.h"
#include "yatfhe/bootstrapping.h"
#include "yatfhe/keyswitching.h"
#include "yatfhe/numeric.h"
#include "yatfhe/tlwe.h"
#include "yatfhe/trgsw.h"

struct ClientDataTE0 {
    vector<Trlwe> valRlwe;

    ClientDataTE0(const int dataSize, const YatfheParameters& param) :
        valRlwe(dataSize, Trlwe{param}){}

    size_t bytes() const {
        if (valRlwe.empty()) return 0;
        const auto& v = valRlwe[0];
        return valRlwe.size() * (v.k + 1) * v.N * sizeof(Torus);
    }
};

struct PlainNodeTE0 {
    int featureIdx{-1};
    int parentIdx{-1};
    int leftChild{-1};
    int rightChild{-1};
    IntPolynomial threshold{};

    PlainNodeTE0() = default;

    explicit PlainNodeTE0(const int N): threshold{N} {}
};

struct DecisionTreeTE0CPC {
    int nodeCount{};
    int featureCount{};
    int quantWidth{};
    std::vector<PlainNodeTE0> nodes;
    std::vector<int> leafParentsIdx;
    std::vector<Integer> leafValues;

    DecisionTreeTE0CPC() = default;

    DecisionTreeTE0CPC(const int nodeCount, const int leafCount, const int features , const YatfheParameters& p) :
        nodeCount(nodeCount),
        featureCount(features),
        nodes(nodeCount, PlainNodeTE0{p.N}),
        leafValues(leafCount){}
};

struct EncNodeTE0CPC {
    Trlwe valLeft;
    Trlwe valRight;

    EncNodeTE0CPC() = default;

    explicit EncNodeTE0CPC(const YatfheParameters& p) :
        valLeft(p),
        valRight(p) {}
};

struct DecisionTreeTE0CPCHelper {
    std::vector<Tlwe> pathSum;
    std::vector<Trlwe> truePathValues;
    std::vector<EncNodeTE0CPC> eNodes;
    std::vector<Trlwe> cumSum;
    std::vector<ScaledTlwe> scaledPathSum;
    std::vector<TorusPolynomial> tv;
    std::vector<Trlwe> leafVals;
    std::vector<Tlwe> tlwes;
    Trlwe accTrlwe;
    Tlwe groupTmp;
    std::vector<std::pair<int,bool>> leafSlots; // (nodeIdx, isLeft)

    DecisionTreeTE0CPCHelper(const DecisionTreeTE0CPC& tree, const YatfheParameters& p) :
        pathSum(tree.leafValues.size(), Tlwe{p.n}),
        truePathValues(tree.leafValues.size(), Trlwe{p}),
        eNodes(tree.nodeCount, EncNodeTE0CPC{p}),
        cumSum(tree.nodeCount, Trlwe{p}),
        scaledPathSum(tree.leafValues.size(), ScaledTlwe{p.N * 2, p.n}),
        tv(tree.leafValues.size(), TorusPolynomial{p.N}),
        leafVals(tree.leafValues.size(), Trlwe{p}),
        tlwes(tree.leafValues.size(), Tlwe{p.N * p.k}),
        accTrlwe(p),
        groupTmp(p.N * p.k),
        leafSlots(tree.leafValues.size()) {
        for (size_t i = 0; i < tree.leafValues.size(); i++) {
            generateTestPolynomialValue(tv[i], tree.leafValues[i]);
        }
        int s = 0;
        for (const auto idx : tree.leafParentsIdx) {
            if (tree.nodes[idx].leftChild  == -1) leafSlots[s++] = {idx, true};
            if (tree.nodes[idx].rightChild == -1) leafSlots[s++] = {idx, false};
        }
    }

    // Delta-scaled variant for GBDT.
    DecisionTreeTE0CPCHelper(const DecisionTreeTE0CPC& tree, const int delta, const YatfheParameters& p) :
        pathSum(tree.leafValues.size(), Tlwe{p.n}),
        truePathValues(tree.leafValues.size(), Trlwe{p}),
        eNodes(tree.nodeCount, EncNodeTE0CPC{p}),
        cumSum(tree.nodeCount, Trlwe{p}),
        scaledPathSum(tree.leafValues.size(), ScaledTlwe{p.N * 2, p.n}),
        tv(tree.leafValues.size(), TorusPolynomial{p.N}),
        leafVals(tree.leafValues.size(), Trlwe{p}),
        tlwes(tree.leafValues.size(), Tlwe{p.N * p.k}),
        accTrlwe(p),
        groupTmp(p.N * p.k),
        leafSlots(tree.leafValues.size()) {
        const int modQ = 2 * p.N;
        const int boundary = modQ / p.torusBase / 2;
        for (size_t i = 0; i < tree.leafValues.size(); i++) {
            const Integer v = tree.leafValues[i];
            for (int j = 0; j < p.N; j++) {
                const Integer tmp = j < boundary ? v : (j >= p.N - boundary ? -v : 0);
                tv[i].coeffs[j] = modSwitchToTorusGeneral(static_cast<int32_t>(tmp), delta, TORUS_Q);
            }
        }
        int s = 0;
        for (const auto idx : tree.leafParentsIdx) {
            if (tree.nodes[idx].leftChild  == -1) leafSlots[s++] = {idx, true};
            if (tree.nodes[idx].rightChild == -1) leafSlots[s++] = {idx, false};
        }
    }
};

void encClientValTE0(ClientDataTE0& clientData, const vector<int>& val, const TrlweKey& key, const YatfheParameters& param);

void buildTE0TreeCPC(DecisionTreeTE0CPC& tree, const vector<Node>& nodes, const YatfheParameters& param);

void rotTE0TreeCPC(DecisionTreeTE0CPCHelper& treeHelper, const DecisionTreeTE0CPC& tree, const ClientDataTE0& clientData,
                         const YatfheParameters& param, int numOuterThreads = 0);

void sumTE0TreePathCPC(DecisionTreeTE0CPCHelper& treeHelper, const DecisionTreeTE0CPC& tree, const TlweKeySwitchingKey& ksk,
                        const YatfheParameters& param, int numOuterThreads = 0);

void calTE0TreeCPC(DecisionTreeTE0CPCHelper& treeHelper, const BootstrappingKeyMP& bskMP, const YatfheParameters& param,
                   int numOuterThreads = 0);

void groupLeafTE0TreeCPC(Tlwe& res, DecisionTreeTE0CPCHelper& treeHelper, const TlweKeySwitchingKey& ksk,
                        const YatfheParameters& param);

void traverseTE0TreeCPC(Tlwe& res, DecisionTreeTE0CPCHelper& treeHelper, const DecisionTreeTE0CPC& tree,
                        const ClientDataTE0& cd, const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk,
                        const YatfheParameters& param,
                        int numOuterThreads = 0);

void buildTE0ForestCPC(std::vector<DecisionTreeTE0CPC>& forest, const std::vector<std::vector<Node>>& nodes,
                       int featureCount, const YatfheParameters& param);

void inferenceTE0GradientForestCPC(std::vector<Tlwe>& out, const std::vector<DecisionTreeTE0CPC>& forest,
                                   const std::vector<int>& treeClass, int numClasses, const std::vector<int>& initQ,
                                   const ClientDataTE0& cd,
                                   const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk, int delta,
                                   const YatfheParameters& param, int numOuterThreads = 0);

void inferenceTE0GradientForestCPCTreeParallel(std::vector<Tlwe>& out, const std::vector<DecisionTreeTE0CPC>& forest,
                                               const std::vector<int>& treeClass, int numClasses,
                                               const std::vector<int>& initQ, const ClientDataTE0& cd,
                                               const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk,
                                               const int delta, const YatfheParameters& param,
                                               int numOuterThreads = 0);

#endif //THETAEQ0_DECISION_TREE_H
