//
// Created by xsong93 on 8/5/26.
//
#ifndef BREAKDOWN_COMMON_H
#define BREAKDOWN_COMMON_H

#include "yatfhe/bootstrapping.h"
#include "yatfhe/keyswitching.h"
#include "yatfhe/tlwe.h"
#include "yatfhe/trgsw.h"
#include "yatfhe/trlwe.h"
#include "yatfhe/yatfhe_parameters.h"
#include "yautil/initializer.h"
#include "yautil/tool.h"

#include "include/v2i_decision_tree.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <optional>
#include <vector>
#include <stdexcept>
#include <nlohmann/json.hpp>


struct ForestData {
    string dataset;
    int maxDepth{};
    int nTrees{};
    int nFeatures{};
    vector<vector<Node>> nodes;
    vector<vector<int>> samples;
    vector<int> expected;
};

inline ForestData loadForestFromJson(const string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open tree: " + path);
    }
    nlohmann::json j;
    f >> j;

    ForestData fd;
    fd.dataset = j.at("dataset").get<string>();
    fd.maxDepth = j.at("maxDepth").get<int>();
    fd.nTrees = j.at("nTrees").get<int>();
    fd.nFeatures = j.at("nFeatures").get<int>();

    for (const auto& tree : j.at("forest")) {
        vector<Node> treeNodes;
        for (const auto& n : tree) {
            treeNodes.push_back(Node{
                n.at("isLeaf").get<bool>(),
                n.at("featureIdx").get<int>(),
                n.at("threshold").get<int>(),
                n.at("parentIdx").get<int>(),
                n.at("leftChild").get<int>(),
                n.at("rightChild").get<int>(),
                n.at("depthOnTree").get<int>(),
                n.at("leafValueLeft").get<int>(),
                n.at("leafValueRight").get<int>(),
                n.at("pathValueLeft").get<int>(),
                n.at("pathValueRight").get<int>(),
                n.at("quantWidth").get<int>(),
            });
        }
        fd.nodes.push_back(std::move(treeNodes));
    }
    for (const auto& row : j.at("samples")) {
        fd.samples.push_back(row.get<vector<int>>());
    }
    fd.expected = j.at("expected").get<vector<int>>();
    return fd;
}

inline int computeLeafCount(const vector<Node>& nodes) {
    int count = 0;
    for (const auto& n : nodes) {
        if (n.leftChild  == -1) count++;
        if (n.rightChild == -1) count++;
    }
    return count;
}

inline string baseName(const string& path) {
    auto pos = path.find_last_of("/\\");
    return (pos == string::npos) ? path : path.substr(pos + 1);
}



struct ParamOverrides {
    int torusBase{-1};   // p, plaintext modulus
    int radixBits{-1};   // B = 2^radixBits, gadget base
    int l{-1};           // full gadget length
    int lApprox{-1};     // approximate (used) gadget length
    int nLwe{-1};        // n
    int nRlwe{-1};       // N
    int lweNoiseB{-1};   // LWE/KSK TUniform bound bits
    int rlweNoiseB{-1};  // RLWE/BSK TUniform bound bits
};

struct BenchOptions {
    vector<string> treePaths;
    string jsonOutputPath;
    int numThreads{1};
    int warmupRotations{0};
    ParamOverrides overrides;
};

inline void printUsage(const char* prog, const char* scheme) {
    cerr << "Usage: " << prog << " <tree.json> [tree2.json ...]\n"
         << "  --json-output <out.json>   write timings to file\n"
         << "  --threads <n>              worker threads (default 1)\n"
         << "  --warmup <r>               untimed blind rotations cycles\n"
         << "Parameter overrides (default: the " << scheme << " set compiled into this binary)\n"
         << "  --torus-base <p>           plaintext modulus\n"
         << "  --radix-bits <b>           gadget base B = 2^b\n"
         << "  --l <l>                    full gadget length\n"
         << "  --l-approx <la>            approximate gadget length\n"
         << "  --n <n>                    LWE dimension\n"
         << "  --N <N>                    RLWE ring dimension\n"
         << "  --lwe-noise-b <b>          LWE/KSK TUniform bound bits\n"
         << "  --rlwe-noise-b <b>         RLWE/BSK TUniform bound bits\n";
}

inline bool parseArgs(int argc, char* argv[], const char* scheme, BenchOptions& opt) {
    if (argc < 2) {
        printUsage(argv[0], scheme);
        return false;
    }

    bool bad = false;
    auto next = [&](int& i, const string& flag) -> const char* {
        if (++i < argc) return argv[i];
        cerr << "Error: " << flag << " requires a value.\n";
        bad = true;
        return nullptr;
    };
    auto setInt = [&](int& i, const string& flag, int& field) {
        const char* raw = next(i, flag);
        if (!raw) return;
        const string s = raw;
        try {
            size_t used = 0;
            const int val = std::stoi(s, &used);
            if (used != s.size()) throw std::invalid_argument("trailing characters");
            field = val;
        } catch (const std::exception&) {
            cerr << "Error: " << flag << " expects an integer, got '" << s << "'.\n";
            bad = true;
        }
    };
    auto setParam = [&](int& i, const string& flag, int ParamOverrides::* field) {
        setInt(i, flag, opt.overrides.*field);
    };
    for (int i = 1; i < argc && !bad; ++i) {
        string arg = argv[i];
        const char* v = nullptr;
        if (arg == "--json-output") { if ((v = next(i, arg))) opt.jsonOutputPath = v; }
        else if (arg == "--threads") setInt(i, arg, opt.numThreads);
        else if (arg == "--warmup") setInt(i, arg, opt.warmupRotations);
        else if (arg == "--torus-base") setParam(i, arg, &ParamOverrides::torusBase);
        else if (arg == "--radix-bits") setParam(i, arg, &ParamOverrides::radixBits);
        else if (arg == "--l") setParam(i, arg, &ParamOverrides::l);
        else if (arg == "--l-approx") setParam(i, arg, &ParamOverrides::lApprox);
        else if (arg == "--n") setParam(i, arg, &ParamOverrides::nLwe);
        else if (arg == "--N") setParam(i, arg, &ParamOverrides::nRlwe);
        else if (arg == "--lwe-noise-b") setParam(i, arg, &ParamOverrides::lweNoiseB);
        else if (arg == "--rlwe-noise-b") setParam(i, arg, &ParamOverrides::rlweNoiseB);
        else if (arg.rfind("--", 0) == 0) {
            cerr << "Error: unknown option " << arg << "\n";
            return false;
        } else { opt.treePaths.push_back(arg); }
    }
    if (bad) return false;
    if (opt.numThreads < 1) opt.numThreads = 1;
    if (opt.warmupRotations < 0) opt.warmupRotations = 0;
    if (opt.treePaths.empty()) {
        cerr << "Error: no tree files specified.\n";
        return false;
    }
    return true;
}

inline void applyOverrides(YatfheParameters& param, const ParamOverrides& ov) {
    if (ov.torusBase > 0) param.torusBase = ov.torusBase;
    if (ov.radixBits > 0) param.setRadixBits(ov.radixBits);
    if (ov.l > 0) param.l = ov.l;
    if (ov.lApprox > 0) param.lApprox = ov.lApprox;
    if (ov.nLwe > 0) param.n = ov.nLwe;
    if (ov.nRlwe > 0) param.N = ov.nRlwe;
    if (ov.lweNoiseB > 0) param.lweNoiseB = ov.lweNoiseB;
    if (ov.rlweNoiseB > 0) param.rlweNoiseB = ov.rlweNoiseB;
}

inline void printParams(const YatfheParameters& param) {
    printf("n:%d, k:%d, N:%d, Torus:%d, b:%d, l:%d, lA:%d, p:%d, lweNoiseB:%d, rlweNoiseB:%d\n",
           param.n, param.k, param.N, param.torusBits, param.radixBits, param.l, param.lApprox, param.torusBase,
           param.lweNoiseB, param.rlweNoiseB);
}

struct EvalKeys {
    TlweKey tlweKey;
    TrgswKey trgswKey;
    BootstrappingKeyMP bskMP;
    TlweKeySwitchingKey ksk;

    EvalKeys(const YatfheParameters& param, const int level) : tlweKey{param}, trgswKey{param}, bskMP{param, level}, ksk{param} {
        genTlweKey(tlweKey);
        genTrlweKey(trgswKey.trlweKey);
        genBootstrappingKeyMP(bskMP, trgswKey, tlweKey, param);
        genTlweKeySwitchingKey(ksk, trgswKey.trlweKey, tlweKey, param);
    }

    TrlweKey& trlweKey() {
        return trgswKey.trlweKey;
    }
};

inline size_t trlweWords(const Trlwe& c) {
    return static_cast<size_t>(c.k + 1) * c.N;
}

inline size_t trgswMPWords(const TrgswMP& g) {
    size_t w = 0;
    for (const auto& row : g.c) {
        for (const auto& c : row) {
            w += trlweWords(c);
        }
    }
    for (const auto& c : g.cPrime) {
        w += trlweWords(c);
    }
    return w;
}

inline size_t bskWords(const BootstrappingKeyMP& bsk) {
    size_t w = 0;
    for (const auto& row : bsk.bsk) {
        for (const auto& g : row) {
            w += trgswMPWords(g);
        }
    }
    return w;
}

inline size_t kskWords(const TlweKeySwitchingKey& ksk) {
    size_t w = 0;
    for (const auto& row : ksk.decomposedKsk) {
        for (const auto& t : row) {
            w += static_cast<size_t>(t.n) + 1;
        }
    }
    return w;
}

inline size_t packedBytes(const size_t words, const int bits) {
    return (words * static_cast<size_t>(bits) + 7) / 8;
}

struct SetupCost {
    double bskMB{}, kskMB{};
    double bskPackedMB{}, kskPackedMB{};
    double totalMB() const { return bskMB + kskMB; }
    double totalPackedMB() const { return bskPackedMB + kskPackedMB; }
};

inline SetupCost measureSetupCost(const BootstrappingKeyMP& bsk, const TlweKeySwitchingKey& ksk,
                                  const YatfheParameters& param) {
    constexpr double MB = 1024.0 * 1024.0;
    const size_t bskW = bskWords(bsk);
    const size_t kskW = kskWords(ksk);
    return SetupCost {
        static_cast<double>(bskW * sizeof(Torus)) / MB,
        static_cast<double>(kskW * sizeof(LweTorus)) / MB,
        static_cast<double>(packedBytes(bskW, param.torusBits)) / MB,
        static_cast<double>(packedBytes(kskW, param.qLweBits)) / MB
    };
}

inline void printSetupCost(const SetupCost& sc, const BootstrappingKeyMP& bsk, const TlweKeySwitchingKey& ksk,
                           const YatfheParameters& param, const int level) {
    cout << "\n--- Setup upload (one-time key material) ---\n"
         << "  BSK  (" << bsk.n << " x " << bsk.bsk[0].size() << " TRGSW, l=" << level << "): "
         << sc.bskMB << " MB stored, " << sc.bskPackedMB << " MB packed@" << param.torusBits << "b\n"
         << "  KSK  (" << ksk.nCurrKey << " x " << ksk.level << " TLWE, n=" << ksk.nTargetKey << "): "
         << sc.kskMB << " MB stored, " << sc.kskPackedMB << " MB packed@" << param.qLweBits << "b\n"
         << "  total: " << sc.totalMB() << " MB stored, " << sc.totalPackedMB() << " MB packed\n";
}

inline nlohmann::json setupToJson(const SetupCost& sc, const BootstrappingKeyMP& bsk, const TlweKeySwitchingKey& ksk,
                                  const YatfheParameters& param, const int level) {
    return {
        {"bsk_MB", sc.bskMB},
        {"ksk_MB", sc.kskMB},
        {"total_MB", sc.totalMB()},
        {"bsk_packed_MB", sc.bskPackedMB},
        {"ksk_packed_MB", sc.kskPackedMB},
        {"total_packed_MB", sc.totalPackedMB()},
        {"bskCount", bsk.n},
        {"bskLevel", level},
        {"kskRows", ksk.nCurrKey},
        {"kskLevel", ksk.level},
        {"torusBits", param.torusBits},
        {"qLweBits", param.qLweBits},
        {"n", param.n},
        {"N", param.N},
        {"k", param.k},
        {"torusBase", param.torusBase},
        {"radixBits", param.radixBits},
        {"l", param.l},
        {"lApprox", param.lApprox},
        {"lweNoiseB", param.lweNoiseB},
        {"rlweNoiseB", param.rlweNoiseB}
    };
}

using BenchClock = std::chrono::steady_clock;
using BenchMs = std::chrono::duration<double, std::milli>;

template <typename Fn>
double measureQuiet(Fn&& fn) {
    auto t0 = BenchClock::now();
    fn();
    return BenchMs(BenchClock::now() - t0).count();
}

inline double processCpuMs() {
    timespec ts{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) / 1e6;
}

struct StageTime {
    double wallMs{0};
    double cpuMs{0};
};

template <typename Fn>
StageTime measureBoth(Fn&& fn) {
    const double c0 = processCpuMs();
    const auto t0 = BenchClock::now();
    fn();
    const double wall = BenchMs(BenchClock::now() - t0).count();
    return {wall, processCpuMs() - c0};
}

inline void writeOutput(const nlohmann::json& output, const string& jsonOutputPath) {
    if (jsonOutputPath.empty()) {
        cout << "\n--- Timing JSON (stdout) ---\n" << output.dump(2) << "\n";
        return;
    }
    std::ofstream of(jsonOutputPath);
    if (!of.is_open()) {
        cerr << "Warning: cannot write JSON output to " << jsonOutputPath << "\n";
    } else {
        of << output.dump(2) << "\n";
        cout << "\nTiming data written to: " << jsonOutputPath << "\n";
    }
}

#endif //BREAKDOWN_COMMON_H
