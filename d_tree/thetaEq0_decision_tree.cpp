//
// Created by xsong93 on 6/15/26.
//

#include <algorithm>
#include <vector>

#include "include/thread_balance.h"
#include "include/v2i_decision_tree.h"
#include "include/thetaEq0_decision_tree.h"

#include "yatfhe/blind_rotate.h"
#include "yatfhe/keyswitching.h"
#include "yatfhe/numeric.h"
#include "yatfhe/trlwe.h"
#include "yautil/multi_threading.h"

void encClientValTE0(ClientDataTE0& clientData, const vector<int>& val, const TrlweKey& key, const YatfheParameters& param) {
    const Torus one = modSwitchToTorus32(1, param.torusBase);
    // enc
    for (auto j = 0; j < val.size(); j++) {
        symEncTrlweSingleSample(clientData.valRlwe[j], key, one, val[j]);
    }
}

void buildTE0TreeCPC(DecisionTreeTE0CPC& tree, const vector<Node>& nodes, const YatfheParameters& param) {
    int leafCount = 0;
    for (int x = 0; x < nodes.size(); x++) {
        const auto& node = nodes[x];
        auto& treeNode = tree.nodes[x];
        treeNode.parentIdx = node.parentIdx;
        treeNode.featureIdx = node.featureIdx;
        treeNode.leftChild = node.leftChild;
        treeNode.rightChild = node.rightChild;

        if (node.leftChild == -1 || node.rightChild == -1) {
            tree.leafParentsIdx.push_back(x);
        }

        for (auto i = param.N - 1; i >= 0; i--) {
            if (i == 0 || i >= param.N - node.threshold) {
                treeNode.threshold.coeffs[i] = 0;
                continue;
            }
            treeNode.threshold.coeffs[i] = -1;
        }

        if (node.isLeaf) {
            for (int j = 0; j < 2; j++) {
                const auto child = j == 0 ? node.leftChild : node.rightChild;

                // leaf with a single child
                if (child != -1) {
                    continue;
                }

                tree.leafValues[leafCount++] = j == 0 ? node.leafValueLeft : node.leafValueRight;
            }
        }
    }
}

void rotTE0TreeCPC(DecisionTreeTE0CPCHelper& treeHelper, const DecisionTreeTE0CPC& tree, const ClientDataTE0& clientData,
                        const YatfheParameters& param, int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;
    const int N = tree.nodeCount;
    const int T = balancedThreadCount(N, numOuterThreads);
    const Torus oneT = modSwitchToTorus32(1, param.torusBase);
    vector<thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; t++) {
        threads.emplace_back([&, t, T, N, oneT] {
            for (int i = t; i < N; i += T) {
                multTrlweWithPolyNtt(treeHelper.eNodes[i].valLeft,
                                     clientData.valRlwe[tree.nodes[i].featureIdx],
                                     tree.nodes[i].threshold, param);
                subTrlweFromConst(treeHelper.eNodes[i].valRight, treeHelper.eNodes[i].valLeft, oneT);
            }
        });
    }
    for (auto& t : threads) t.join();
}

void sumTE0TreePathCPC(DecisionTreeTE0CPCHelper& treeHelper, const DecisionTreeTE0CPC& tree, const TlweKeySwitchingKey& ksk,
                        const YatfheParameters& param, int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;
    auto& cumSum = treeHelper.cumSum;
    for (int i = 1; i < tree.nodeCount; i++) {
        const auto parentIdx = tree.nodes[i].parentIdx;
        const auto& currVal = tree.nodes[parentIdx].leftChild == i
                              ? treeHelper.eNodes[parentIdx].valLeft
                              : treeHelper.eNodes[parentIdx].valRight;
        addTrlwe(cumSum[i], cumSum[parentIdx], currVal);
    }

    const int P = static_cast<int>(treeHelper.pathSum.size());
    const int T = balancedThreadCount(P, numOuterThreads);
    vector<thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; t++) {
        threads.emplace_back([&, t, T] {
            for (int s = t; s < P; s += T) {
                const auto [idx, isLeft] = treeHelper.leafSlots[s];
                auto& leafVal = treeHelper.leafVals[s];
                addTrlwe(leafVal, cumSum[idx],
                         isLeft ? treeHelper.eNodes[idx].valLeft
                                : treeHelper.eNodes[idx].valRight);
                auto& tmp = treeHelper.tlwes[s];
                extractTlweFromTrlwe(tmp, leafVal, 0);
                switchKeyForTlwe(treeHelper.pathSum[s], ksk, tmp, param);
            }
        });
    }
    for (auto& t : threads) t.join();
}


void calTE0TreeCPC(DecisionTreeTE0CPCHelper& treeHelper, const BootstrappingKeyMP& bskMP, const YatfheParameters& param,
                   int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;
    const int P = static_cast<int>(treeHelper.pathSum.size());
    const int T = balancedThreadCount(P, numOuterThreads);
    vector<thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; t++) {
        threads.emplace_back([&, t, T] {
            for (int s = t; s < P; s += T) {
                auto& scaledInput = treeHelper.scaledPathSum[s];
                rescaleTlweToNewMod(scaledInput, treeHelper.pathSum[s]);
                genNoiselessTrlweSample(treeHelper.truePathValues[s], treeHelper.tv[s], scaledInput);
                blindRotateMP21Ntt(treeHelper.truePathValues[s], bskMP.bskDft, scaledInput, param);
            }
        });
    }
    for (auto& t : threads) t.join();
}

void groupLeafTE0TreeCPC(Tlwe& res, DecisionTreeTE0CPCHelper& treeHelper, const TlweKeySwitchingKey& ksk,
                        const YatfheParameters& param) {
    auto& acc = treeHelper.accTrlwe;
    clearTrlwe(acc);
    for (const auto& truePathValue : treeHelper.truePathValues) {
        accumulateTrlwe(acc, truePathValue);
    }
    auto& tmp = treeHelper.groupTmp;
    extractTlweFromTrlwe(tmp, acc, 0);
    switchKeyForTlwe(res, ksk, tmp, param);
}

void traverseTE0TreeCPC(Tlwe& res, DecisionTreeTE0CPCHelper& treeHelper, const DecisionTreeTE0CPC& tree,
                        const ClientDataTE0& cd, const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk,
                        const YatfheParameters& param, int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;

    // Stage 1: node compare
    {
        const int N = tree.nodeCount;
        const int T = balancedThreadCount(N, numOuterThreads);
        const Torus oneT = modSwitchToTorus32(1, param.torusBase);
        vector<thread> threads;
        threads.reserve(T);
        for (int t = 0; t < T; t++) {
            threads.emplace_back([&, t, T, N, oneT] {
                for (int i = t; i < N; i += T) {
                    multTrlweWithPolyNtt(treeHelper.eNodes[i].valLeft,
                                        cd.valRlwe[tree.nodes[i].featureIdx],
                                        tree.nodes[i].threshold, param);
                    subTrlweFromConst(treeHelper.eNodes[i].valRight,
                                     treeHelper.eNodes[i].valLeft, oneT);
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    // Stage 2 path sum
    auto& cumSum = treeHelper.cumSum;
    for (int i = 1; i < tree.nodeCount; i++) {
        const auto parentIdx = tree.nodes[i].parentIdx;
        const auto& currVal = tree.nodes[parentIdx].leftChild == i
                              ? treeHelper.eNodes[parentIdx].valLeft
                              : treeHelper.eNodes[parentIdx].valRight;
        addTrlwe(cumSum[i], cumSum[parentIdx], currVal);
    }

    // Stages 3 leaf select
    const int P = static_cast<int>(treeHelper.pathSum.size());
    {
        const int T = balancedThreadCount(P, numOuterThreads);
        vector<thread> threads;
        threads.reserve(T);
        for (int t = 0; t < T; t++) {
            threads.emplace_back([&, t, T] {
                for (int s = t; s < P; s += T) {
                    const auto [idx, isLeft] = treeHelper.leafSlots[s];
                    const auto& edgeVal = isLeft ? treeHelper.eNodes[idx].valLeft
                                                 : treeHelper.eNodes[idx].valRight;
                    auto& leafVal = treeHelper.leafVals[s];
                    addTrlwe(leafVal, cumSum[idx], edgeVal);
                    auto& tlweTmp = treeHelper.tlwes[s];
                    extractTlweFromTrlwe(tlweTmp, leafVal, 0);
                    switchKeyForTlwe(treeHelper.pathSum[s], ksk, tlweTmp, param);
                    auto& scaledInput = treeHelper.scaledPathSum[s];
                    rescaleTlweToNewMod(scaledInput, treeHelper.pathSum[s]);
                    genNoiselessTrlweSample(treeHelper.truePathValues[s], treeHelper.tv[s], scaledInput);
                    blindRotateMP21Ntt(treeHelper.truePathValues[s], bskMP.bskDft, scaledInput, param);
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    // Stage 4: leaf group
    groupLeafTE0TreeCPC(res, treeHelper, ksk, param);
}

void buildTE0ForestCPC(std::vector<DecisionTreeTE0CPC>& forest, const std::vector<std::vector<Node>>& nodes,
                       const int featureCount, const YatfheParameters& param) {
    forest.reserve(nodes.size());
    for (const auto& treeNodes : nodes) {
        const int nodeCount = static_cast<int>(treeNodes.size());
        int leafCount = 0;
        for (const auto& n : treeNodes) {
            if (n.leftChild  == -1) leafCount++;
            if (n.rightChild == -1) leafCount++;
        }
        forest.emplace_back(nodeCount, leafCount, featureCount, param);
        buildTE0TreeCPC(forest.back(), treeNodes, param);
    }
}

void inferenceTE0GradientForestCPC(std::vector<Tlwe>& out, const std::vector<DecisionTreeTE0CPC>& forest,
                                   const std::vector<int>& treeClass, int numClasses, const std::vector<int>& initQ,
                                   const ClientDataTE0& cd,
                                   const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk, const int delta,
                                   const YatfheParameters& param, int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;
    std::vector<Trlwe> acc;
    acc.reserve(numClasses);
    for (int k = 0; k < numClasses; k++) {
        acc.emplace_back(param);
        clearTrlwe(acc.back());
    }

    for (int i = 0; i < forest.size(); i++) {
        const auto& treeI = forest[i];
        DecisionTreeTE0CPCHelper treeHelper{treeI, delta, param};
        rotTE0TreeCPC(treeHelper, treeI, cd, param, numOuterThreads);
        sumTE0TreePathCPC(treeHelper, treeI, ksk, param, numOuterThreads);
        calTE0TreeCPC(treeHelper, bskMP, param, numOuterThreads);

        for (const auto& truePathValue : treeHelper.truePathValues) {
            accumulateTrlwe(acc[treeClass[i]], truePathValue);
        }
    }

    const bool addInit = static_cast<int>(initQ.size()) == numClasses;
    out.clear();
    out.reserve(numClasses);
    for (int k = 0; k < numClasses; k++) {
        Tlwe tmp{param.k * param.N};
        extractTlweFromTrlwe(tmp, acc[k], 0);
        out.emplace_back(param.n);
        switchKeyForTlwe(out.back(), ksk, tmp, param);
        if (addInit)
            out.back().b = addTorus(LWE_Q, out.back().b, modSwitchToTorusGeneral(initQ[k], delta, LWE_Q));
    }
}

void inferenceTE0GradientForestCPCTreeParallel(std::vector<Tlwe>& out, const std::vector<DecisionTreeTE0CPC>& forest,
                                               const std::vector<int>& treeClass, int numClasses,
                                               const std::vector<int>& initQ, const ClientDataTE0& cd,
                                               const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk,
                                               const int delta, const YatfheParameters& param, int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;
    const int nTrees = static_cast<int>(forest.size());
    const int T = balancedThreadCount(nTrees, numOuterThreads);
    if (T <= 0) return;

    std::vector<std::vector<Trlwe>> localAcc;
    localAcc.reserve(T);
    for (int t = 0; t < T; t++) {
        localAcc.emplace_back();
        localAcc.back().reserve(numClasses);
        for (int k = 0; k < numClasses; k++) {
            localAcc.back().emplace_back(param);
            clearTrlwe(localAcc.back().back());
        }
    }

    {
        vector<thread> threads;
        threads.reserve(T);
        for (int t = 0; t < T; t++) {
            threads.emplace_back([&, t, T] {
                auto& myAcc = localAcc[t];
                for (int i = t; i < nTrees; i += T) {
                    const auto& treeI = forest[i];
                    DecisionTreeTE0CPCHelper treeHelper{treeI, delta, param};   // delta-scaled leaf encoding
                    rotTE0TreeCPC(treeHelper, treeI, cd, param, 1);
                    sumTE0TreePathCPC(treeHelper, treeI, ksk, param, 1);
                    calTE0TreeCPC(treeHelper, bskMP, param, 1);
                    for (const auto& truePathValue : treeHelper.truePathValues)
                        accumulateTrlwe(myAcc[treeClass[i]], truePathValue);
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    std::vector<Trlwe> acc;
    acc.reserve(numClasses);
    for (int k = 0; k < numClasses; k++) {
        acc.emplace_back(param);
        clearTrlwe(acc.back());
        for (int t = 0; t < T; t++)
            addTrlwe(acc[k], acc[k], localAcc[t][k]);
    }

    const bool addInit = static_cast<int>(initQ.size()) == numClasses;
    out.clear();
    out.reserve(numClasses);
    for (int k = 0; k < numClasses; k++) {
        Tlwe tmp{param.k * param.N};
        extractTlweFromTrlwe(tmp, acc[k], 0);
        out.emplace_back(param.n);
        switchKeyForTlwe(out.back(), ksk, tmp, param);
        if (addInit)
            out.back().b = addTorus(LWE_Q, out.back().b, modSwitchToTorusGeneral(initQ[k], delta, LWE_Q));
    }
}