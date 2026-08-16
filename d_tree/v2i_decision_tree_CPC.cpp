//
// Created by Xintong Song on 2026/5/29.
//

#include <vector>

#include "include/thread_balance.h"
#include "include/v2i_decision_tree.h"
#include "include/v2i_decision_tree_CPC.h"

#include "yatfhe/blind_rotate.h"
#include "yatfhe/keyswitching.h"
#include "yatfhe/numeric.h"
#include "yatfhe/trlwe.h"
#include "yautil/multi_threading.h"

void encClientValV2I(ClientData& clientData, const vector<int>& val, const TrlweKey& key, const YatfheParameters& param) {
    const Torus one = modSwitchToTorus32(1, param.torusBase);
    // enc
    for (auto j = 0; j < val.size(); j++) {
        symEncTrlweSingleSample(clientData.valRlwe[j], key, one, val[j]);
    }
}

void buildV2ITreeCPC(DecisionTreeV2ICPC& tree, const vector<Node>& nodes, const YatfheParameters& param) {
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
            if (i == 0) {
                treeNode.threshold.coeffs[i] = x == 0 ? -1 : -1 << (node.depthOnTree - 1);
                continue;
            }
            if (i >= param.N - node.threshold) {
                treeNode.threshold.coeffs[i] = x == 0 ? 1 : 1 << (node.depthOnTree - 1);
                continue;
            }
            treeNode.threshold.coeffs[i] = x == 0 ? 0 : -1 << (node.depthOnTree - 1);
        }

        if (node.isLeaf) {
            for (int j = 0; j < 2; j++) {
                const auto child = j == 0 ? node.leftChild : node.rightChild;

                // leaf with a single child
                if (child != -1) {
                    continue;
                }

                const auto leafValue = j == 0 ? node.leafValueLeft : node.leafValueRight;
                const auto pathValue = j == 0 ? node.pathValueLeft : node.pathValueRight;
                const int sign = pathValue < 0 ? -1 : 1;
                const auto valuePos = pathValue < 0 ? param.N + tree.bucketSize * pathValue : tree.bucketSize * pathValue;
                tree.leafSlots.emplace_back(valuePos, leafValue * sign);
            }
        }
    }
}

void rotV2ITreeCPC(DecisionTreeV2ICPCHelper& treeHelper, const DecisionTreeV2ICPC& tree, const ClientData& clientData,
                        const YatfheParameters& param, int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;
    const int N = tree.nodeCount;
    const int T = balancedThreadCount(N, numOuterThreads);
    vector<thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; t++) {
        threads.emplace_back([&, t, T, N] {
            for (int i = t; i < N; i += T) {
                multTrlweWithPolyNtt(treeHelper.nodeVal[i], clientData.valRlwe[tree.nodes[i].featureIdx],
                                     tree.nodes[i].threshold, param);
            }
        });
    }
    for (auto& t : threads) t.join();
}

void sumV2ITreePathCPC(DecisionTreeV2ICPCHelper& treeHelper, const DecisionTreeV2ICPC& tree, const TlweKeySwitchingKey& ksk,
                       const YatfheParameters& param, int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;
    auto& cumSum = treeHelper.cumSum;
    addTrlwe(cumSum[0], treeHelper.nodeVal[0]);
    for (int i = 1; i < tree.nodeCount; i++) {
        addTrlwe(cumSum[i], cumSum[tree.nodes[i].parentIdx], treeHelper.nodeVal[i]);
    }
    const int P = static_cast<int>(tree.leafParentsIdx.size());
    const int T = balancedThreadCount(P, numOuterThreads);
    vector<thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; t++) {
        threads.emplace_back([&, t, T] {
            for (int s = t; s < P; s += T) {
                auto& tmp = treeHelper.tlwes[s];
                extractTlweFromTrlwe(tmp, cumSum[tree.leafParentsIdx[s]], 0);
                switchKeyForTlwe(treeHelper.pathSum[s], ksk, tmp, param);
            }
        });
    }
    for (auto& t : threads) t.join();
}

void calV2ITreeCPC(DecisionTreeV2ICPCHelper& treeHelper, const BootstrappingKeyMP& bskMP, const YatfheParameters& param,
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
                genNoiselessTrlweSample(treeHelper.indexRange[s], treeHelper.tvs[s], scaledInput);
                blindRotateMP21Ntt(treeHelper.indexRange[s], bskMP.bskDft, scaledInput, param);
            }
        });
    }
    for (auto& t : threads) t.join();
}

void groupLeafV2ITreeCPC(Tlwe& res, DecisionTreeV2ICPCHelper& treeHelper, const TlweKeySwitchingKey& ksk,
                         const YatfheParameters& param) {
    auto& acc = treeHelper.accTrlwe;
    clearTrlwe(acc);
    for (auto& truePathValue : treeHelper.indexRange) {
        accumulateTrlwe(acc, truePathValue);
    }
    auto& tmp = treeHelper.groupTmp;
    extractTlweFromTrlwe(tmp, acc, 0);
    switchKeyForTlwe(res, ksk, tmp, param);
}

void traverseV2ITreeCPC(Tlwe& res, DecisionTreeV2ICPCHelper& treeHelper, const DecisionTreeV2ICPC& tree,
                            const ClientData& cd, const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk,
                            const YatfheParameters& param, int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;

    // Stage 1: node compare
    rotV2ITreeCPC(treeHelper, tree, cd, param, numOuterThreads);

    // Stage 2 path sum
    auto& cumSum = treeHelper.cumSum;
    addTrlwe(cumSum[0], treeHelper.nodeVal[0]);
    for (int i = 1; i < tree.nodeCount; i++) {
        addTrlwe(cumSum[i], cumSum[tree.nodes[i].parentIdx], treeHelper.nodeVal[i]);
    }

    // Stages 3 leaf select
    const auto& leafIdx = tree.leafParentsIdx;
    const int P = static_cast<int>(leafIdx.size());
    const int T = balancedThreadCount(P, numOuterThreads);
    {
        vector<thread> threads;
        threads.reserve(T);
        for (int t = 0; t < T; t++) {
            threads.emplace_back([&, t, T] {
                for (int i = t; i < P; i += T) {
                    const auto idx = leafIdx[i];
                    auto& tmp = treeHelper.tlwes[i];
                    extractTlweFromTrlwe(tmp, cumSum[idx], 0);
                    switchKeyForTlwe(treeHelper.pathSum[i], ksk, tmp, param);
                    auto& scaledInput = treeHelper.scaledPathSum[i];
                    rescaleTlweToNewMod(scaledInput, treeHelper.pathSum[i]);
                    genNoiselessTrlweSample(treeHelper.indexRange[i], treeHelper.tvs[i], scaledInput);
                    blindRotateMP21Ntt(treeHelper.indexRange[i], bskMP.bskDft, scaledInput, param);
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    // Stages 4 leaf group
    groupLeafV2ITreeCPC(res, treeHelper, ksk, param);
}

void traverseV2ITreeNoSwitchCPC(DecisionTreeV2ICPCHelper& treeHelper, const DecisionTreeV2ICPC& tree,
                               const ClientData& cd, const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk,
                               const YatfheParameters& param, int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;
    rotV2ITreeCPC(treeHelper, tree, cd, param, numOuterThreads);
    sumV2ITreePathCPC(treeHelper, tree, ksk, param, numOuterThreads);
    calV2ITreeCPC(treeHelper, bskMP, param, numOuterThreads);

    auto& acc = treeHelper.accTrlwe;
    clearTrlwe(acc);
    for (auto& truePathValue : treeHelper.indexRange) {
        accumulateTrlwe(acc, truePathValue);
    }
}

void buildV2IForestCPC(std::vector<DecisionTreeV2ICPC>& forest, const std::vector<std::vector<Node>>& nodes,
                       const int featureCount, const YatfheParameters& param) {
    forest.reserve(nodes.size());
    for (const auto& treeNodes : nodes) {
        const int nodeCount = static_cast<int>(treeNodes.size());
        forest.emplace_back(nodeCount, featureCount, param);
        buildV2ITreeCPC(forest.back(), treeNodes, param);
    }
}

void inferenceV2IGradientForestCPC(Tlwe& out, const std::vector<DecisionTreeV2ICPC>& forest, const ClientData& cd,
                                   const BootstrappingKeyMP& bskMP, const TlweKeySwitchingKey& ksk, const int delta,
                                   int initQ, const YatfheParameters& param, int numOuterThreads) {
    if (numOuterThreads <= 0) numOuterThreads = 1;
    Trlwe acc{param};
    for (int i = 0; i < forest.size(); i++) {
        const auto& treeI = forest[i];
        DecisionTreeV2ICPCHelper treeHelper{treeI, delta, param};
        traverseV2ITreeNoSwitchCPC(treeHelper, treeI, cd, bskMP, ksk, param, numOuterThreads); // traverse forest
        addTrlwe(acc, acc, treeHelper.accTrlwe);// result aggregation
    }

    Tlwe tmp{param.k * param.N};
    extractTlweFromTrlwe(tmp, acc, 0);
    switchKeyForTlwe(out, ksk, tmp, param);
    if (initQ != 0) out.b = addTorus(LWE_Q, out.b, modSwitchToTorusGeneral(initQ, delta, LWE_Q));
}

void inferenceV2IGradientForestCPC(std::vector<Tlwe>& out, const std::vector<DecisionTreeV2ICPC>& forest,
                                   const std::vector<int>& treeClass, int numClasses, const std::vector<int>& initQ,
                                   const ClientData& cd,
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
        DecisionTreeV2ICPCHelper treeHelper{treeI, delta, param};
        traverseV2ITreeNoSwitchCPC(treeHelper, treeI, cd, bskMP, ksk, param, numOuterThreads); // traverse forest
        addTrlwe(acc[treeClass[i]], acc[treeClass[i]], treeHelper.accTrlwe);// per-class aggregation
    }

    const bool addInit = static_cast<int>(initQ.size()) == numClasses;
    out.clear();
    out.reserve(numClasses);
    for (int k = 0; k < numClasses; k++) {
        Tlwe tmp{param.k * param.N};
        extractTlweFromTrlwe(tmp, acc[k], 0);
        out.emplace_back(param.n);
        switchKeyForTlwe(out.back(), ksk, tmp, param);
        if (addInit) out.back().b = addTorus(LWE_Q, out.back().b, modSwitchToTorusGeneral(initQ[k], delta, LWE_Q));
    }
}

void inferenceV2IGradientForestCPCTreeParallel(std::vector<Tlwe>& out, const std::vector<DecisionTreeV2ICPC>& forest,
                                               const std::vector<int>& treeClass, int numClasses,
                                               const std::vector<int>& initQ, const ClientData& cd,
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
                    DecisionTreeV2ICPCHelper treeHelper{treeI, delta, param};
                    traverseV2ITreeNoSwitchCPC(treeHelper, treeI, cd, bskMP, ksk, param, 1);
                    addTrlwe(myAcc[treeClass[i]], myAcc[treeClass[i]], treeHelper.accTrlwe);
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
        for (int t = 0; t < T; t++) {
            addTrlwe(acc[k], acc[k], localAcc[t][k]);
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
        if (addInit) out.back().b = addTorus(LWE_Q, out.back().b, modSwitchToTorusGeneral(initQ[k], delta, LWE_Q));
    }
}
