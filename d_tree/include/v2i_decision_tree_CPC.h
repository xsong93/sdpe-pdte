//
// Created by Xintong Song on 2026/5/29.
//
#ifndef V2I_DECISION_TREE_CPC_H
#define V2I_DECISION_TREE_CPC_H

#include <cstddef>
#include <vector>

#include "v2i_decision_tree.h"
#include "yatfhe/bootstrapping.h"
#include "yatfhe/keyswitching.h"
#include "yatfhe/tlwe.h"
#include "yatfhe/trgsw.h"

struct ClientData {
    vector<Trlwe> valRlwe;

    ClientData(const int dataSize, const YatfheParameters& param) :
        valRlwe(dataSize, Trlwe{param}){}

    size_t bytes() const {
        if (valRlwe.empty()) return 0;
        const auto& v = valRlwe[0];
        return valRlwe.size() * (v.k + 1) * v.N * sizeof(Torus);
    }
};

struct CommOverhead {
    size_t upload{};
    size_t download{};
    size_t total() const { return upload + download; }
};

struct PlainNode {
    int featureIdx{-1};
    int parentIdx{-1};
    int leftChild{-1};
    int rightChild{-1};
    IntPolynomial threshold{};

    PlainNode() = default;

    explicit PlainNode(const int N): threshold{N} {}
};

struct DecisionTreeV2ICPC {
    int nodeCount{};
    int featureCount{};
    int bucketSize{};
    std::vector<PlainNode> nodes;
    std::vector<int> leafParentsIdx;
    std::vector<std::pair<int,int>> leafSlots;

    DecisionTreeV2ICPC() = default;

    DecisionTreeV2ICPC(const int nodeCount, const int features , const YatfheParameters& p) :
        nodeCount(nodeCount),
        featureCount(features),
        bucketSize(2 * p.N / p.torusBase),
        nodes(nodeCount, PlainNode{p.N}){}
};

struct DecisionTreeV2ICPCHelper {
    std::vector<Tlwe> pathSum;
    std::vector<Trlwe> indexRange;
    std::vector<Trlwe> nodeVal;
    std::vector<Trlwe> cumSum;
    std::vector<ScaledTlwe> scaledPathSum;
    std::vector<TorusPolynomial> tvs;
    std::vector<Tlwe> tlwes;
    Trlwe accTrlwe;
    Tlwe groupTmp;
    std::vector<int> leafValStart;

    DecisionTreeV2ICPCHelper(const DecisionTreeV2ICPC& tree, const YatfheParameters& p) :
        pathSum(tree.leafParentsIdx.size(), Tlwe{p.n}),
        indexRange(tree.leafParentsIdx.size(), Trlwe{p}),
        nodeVal(tree.nodeCount, Trlwe{p}),
        cumSum(tree.nodeCount, Trlwe{p}),
        scaledPathSum(tree.leafParentsIdx.size(), ScaledTlwe{p.N * 2, p.n}),
        tvs(tree.leafParentsIdx.size(), TorusPolynomial{p.N}),
        tlwes(tree.leafParentsIdx.size(), Tlwe{p.N * p.k}),
        accTrlwe(p),
        groupTmp(p.N * p.k),
        leafValStart(tree.leafParentsIdx.size()) {
            const int modP = p.torusBase;
            const int modQ = 2 * p.N;
            const int boundary = modQ / modP / 2;
            int k = 0;
            for (size_t i = 0; i < tree.leafParentsIdx.size(); i++) {
                leafValStart[i] = k;
                const auto idx = tree.leafParentsIdx[i];
                if (tree.nodes[idx].leftChild  == -1) {
                    int upperBound = tree.leafSlots[k].first == 0 ? boundary : tree.leafSlots[k].first + boundary;
                    int lowerBound = tree.leafSlots[k].first == 0 ? p.N - boundary - 1 : tree.leafSlots[k].first - boundary;
                    for (auto j = 0; j < p.N; j++) {
                        if (tree.leafSlots[k].first == 0) {
                            const auto tmp = j < upperBound ? tree.leafSlots[k].second : j >= lowerBound ? -tree.leafSlots[k].second : 0;
                            tvs[i].coeffs[j] += modSwitchToTorus32(tmp, modP);
                            continue;
                        }
                        const auto tmp = (j >= lowerBound && j < upperBound) ? tree.leafSlots[k].second : 0;
                        tvs[i].coeffs[j] += modSwitchToTorus32(tmp, modP);
                    }
                    k++;
                }
                if (tree.nodes[idx].rightChild == -1) {
                    int upperBound = tree.leafSlots[k].first == 0 ? boundary : tree.leafSlots[k].first + boundary;
                    int lowerBound = tree.leafSlots[k].first == 0 ? p.N - boundary - 1 : tree.leafSlots[k].first - boundary;
                    for (auto j = 0; j < p.N; j++) {
                        if (tree.leafSlots[k].first == 0) {
                            const auto tmp = j < upperBound ? tree.leafSlots[k].second : j >= lowerBound ? -tree.leafSlots[k].second : 0;
                            tvs[i].coeffs[j] += modSwitchToTorus32(tmp, modP);
                            continue;
                        }
                        const auto tmp = (j >= lowerBound && j < upperBound) ? tree.leafSlots[k].second : 0;
                        tvs[i].coeffs[j] += modSwitchToTorus32(tmp, modP);
                    }
                    k++;
                }
            }
        }

        DecisionTreeV2ICPCHelper(const DecisionTreeV2ICPC& tree, const int delta, const YatfheParameters& p) :
        pathSum(tree.leafParentsIdx.size(), Tlwe{p.n}),
        indexRange(tree.leafParentsIdx.size(), Trlwe{p}),
        nodeVal(tree.nodeCount, Trlwe{p}),
        cumSum(tree.nodeCount, Trlwe{p}),
        scaledPathSum(tree.leafParentsIdx.size(), ScaledTlwe{p.N * 2, p.n}),
        tvs(tree.leafParentsIdx.size(), TorusPolynomial{p.N}),
        tlwes(tree.leafParentsIdx.size(), Tlwe{p.N * p.k}),
        accTrlwe(p),
        groupTmp(p.N * p.k),
        leafValStart(tree.leafParentsIdx.size()) {
            const int modP = p.torusBase;
            const int modQ = 2 * p.N;
            const int boundary = modQ / modP / 2;
            int k = 0;
            for (size_t i = 0; i < tree.leafParentsIdx.size(); i++) {
                leafValStart[i] = k;
                const auto idx = tree.leafParentsIdx[i];
                if (tree.nodes[idx].leftChild  == -1) {
                    int upperBound = tree.leafSlots[k].first == 0 ? boundary : tree.leafSlots[k].first + boundary;
                    int lowerBound = tree.leafSlots[k].first == 0 ? p.N - boundary - 1 : tree.leafSlots[k].first - boundary;
                    for (auto j = 0; j < p.N; j++) {
                        if (tree.leafSlots[k].first == 0) {
                            const auto tmp = j < upperBound ? tree.leafSlots[k].second : j >= lowerBound ? -tree.leafSlots[k].second : 0;
                            tvs[i].coeffs[j] += modSwitchToTorusGeneral(tmp, delta, TORUS_Q);
                            continue;
                        }
                        const auto tmp = (j >= lowerBound && j < upperBound) ? tree.leafSlots[k].second : 0;
                        tvs[i].coeffs[j] += modSwitchToTorusGeneral(tmp, delta, TORUS_Q);
                    }
                    k++;
                }
                if (tree.nodes[idx].rightChild == -1) {
                    int upperBound = tree.leafSlots[k].first == 0 ? boundary : tree.leafSlots[k].first + boundary;
                    int lowerBound = tree.leafSlots[k].first == 0 ? p.N - boundary - 1 : tree.leafSlots[k].first - boundary;
                    for (auto j = 0; j < p.N; j++) {
                        if (tree.leafSlots[k].first == 0) {
                            const auto tmp = j < upperBound ? tree.leafSlots[k].second : j >= lowerBound ? -tree.leafSlots[k].second : 0;
                            tvs[i].coeffs[j] += modSwitchToTorusGeneral(tmp, delta, TORUS_Q);
                            continue;
                        }
                        const auto tmp = (j >= lowerBound && j < upperBound) ? tree.leafSlots[k].second : 0;
                        tvs[i].coeffs[j] += modSwitchToTorusGeneral(tmp, delta, TORUS_Q);
                    }
                    k++;
                }
            }
        }
};

void encClientValV2I(ClientData& clientData, const vector<int>& val, const TrlweKey& key, const YatfheParameters& param);

void buildV2ITreeCPC(DecisionTreeV2ICPC& tree, const vector<Node>& nodes, const YatfheParameters& param);

void rotV2ITreeCPC(DecisionTreeV2ICPCHelper& treeHelper, const DecisionTreeV2ICPC& tree, const ClientData& clientData,
                   const YatfheParameters& param, int numOuterThreads = 0);

void sumV2ITreePathCPC(DecisionTreeV2ICPCHelper& treeHelper, const DecisionTreeV2ICPC& tree, const TlweKeySwitchingKey& ksk,
                       const YatfheParameters& param, int numOuterThreads = 0);

void calV2ITreeCPC(DecisionTreeV2ICPCHelper& treeHelper, const BootstrappingKeyMP& bskMP, const YatfheParameters& param,
                   int numOuterThreads = 0);

void groupLeafV2ITreeCPC(Tlwe& res, DecisionTreeV2ICPCHelper& treeHelper, const TlweKeySwitchingKey& ksk,
                         const YatfheParameters& param);

void traverseV2ITreeCPC(Tlwe& res, DecisionTreeV2ICPCHelper& treeHelper, const DecisionTreeV2ICPC& tree,
                            const ClientData& cd, const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk,
                            const YatfheParameters& param, int numOuterThreads = 0);

void traverseV2ITreeNoSwitchCPC(DecisionTreeV2ICPCHelper& treeHelper, const DecisionTreeV2ICPC& tree,
                               const ClientData& cd, const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk,
                               const YatfheParameters& param, int numOuterThreads = 0);

void buildV2IForestCPC(std::vector<DecisionTreeV2ICPC>& forest, const std::vector<std::vector<Node>>& nodes,
                       int featureCount, const YatfheParameters& param);

// Single-class (binary) GBDT
void inferenceV2IGradientForestCPC(Tlwe& out, const std::vector<DecisionTreeV2ICPC>& forest, const ClientData& cd,
                                   const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk, const int delta,
                                   int initQ, const YatfheParameters& param, int numOuterThreads = 0);

// Multiclass GBDT
void inferenceV2IGradientForestCPC(std::vector<Tlwe>& out, const std::vector<DecisionTreeV2ICPC>& forest,
                                   const std::vector<int>& treeClass, int numClasses, const std::vector<int>& initQ,
                                   const ClientData& cd,
                                   const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk, const int delta,
                                   const YatfheParameters& param, int numOuterThreads = 0);

void inferenceV2IGradientForestCPCTreeParallel(std::vector<Tlwe>& out, const std::vector<DecisionTreeV2ICPC>& forest,
                                               const std::vector<int>& treeClass, int numClasses,
                                               const std::vector<int>& initQ, const ClientData& cd,
                                               const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk,
                                               const int delta, const YatfheParameters& param,
                                               int numOuterThreads = 0);

#endif // V2I_DECISION_TREE_CPC_H
