//
// Created by xsong93 on 7/18/26.
//
#include "yatfhe/bootstrapping.h"
#include "include/v2i_decision_tree.h"
#include "include/v2i_decision_tree_CPC.h"
#include "yatfhe/tlwe.h"
#include "yatfhe/trlwe.h"
#include "yatfhe/trgsw.h"
#include "yatfhe/keyswitching.h"
#include "yatfhe/yatfhe_parameters.h"
#include "yautil/initializer.h"
#include "yautil/tool.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <thread>
#include <nlohmann/json.hpp>

#include "include/thetaEq0_decision_tree.h"

struct ForestData {
    string dataset;
    string model{"forest"};
    int maxDepth{};
    int nTrees{};
    int nFeatures{};
    int kappa{8};
    double plaintextAccuracy{};
    double quantizedAccuracy{};
    vector<int> treeClass; // tree i -> class index (GBDT), empty for single-class
    vector<int> initQ; // per-class quantized init score
    vector<vector<Node>> nodes;
    vector<vector<int>> samples;
    vector<int> expected;
};

static ForestData loadForestFromJson(const string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    nlohmann::json j;
    f >> j;

    ForestData fd;
    fd.dataset = j.at("dataset").get<string>();
    fd.maxDepth = j.at("maxDepth").get<int>();
    fd.nTrees = j.at("nTrees").get<int>();
    fd.nFeatures = j.at("nFeatures").get<int>();
    fd.plaintextAccuracy = j.at("plaintextAccuracy").get<double>();
    fd.quantizedAccuracy = j.at("quantizedAccuracy").get<double>();

    if (j.contains("model")) fd.model = j.at("model").get<string>();
    if (j.contains("kappa")) fd.kappa = j.at("kappa").get<int>();
    if (j.contains("treeClass")) fd.treeClass = j.at("treeClass").get<vector<int>>();
    if (j.contains("initQ")) fd.initQ = j.at("initQ").get<vector<int>>();

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
    for (const auto& row : j.at("samples"))
        fd.samples.push_back(row.get<vector<int>>());
    fd.expected = j.at("expected").get<vector<int>>();
    return fd;
}

static string baseName(const string& path) {
    auto pos = path.find_last_of("/\\");
    return (pos == string::npos) ? path : path.substr(pos + 1);
}

static string withGSuffix(const string& path) {
    if (path.empty()) return path;
    auto slash = path.find_last_of("/\\");
    auto nameStart = (slash == string::npos) ? 0 : slash + 1;
    auto dot = path.find_last_of('.');
    auto stemEnd = (dot == string::npos || dot < nameStart) ? path.size() : dot;
    const string stem = path.substr(nameStart, stemEnd - nameStart);
    if (stem.size() >= 2 && stem.compare(stem.size() - 2, 2, "_g") == 0) return path;
    return path.substr(0, stemEnd) + "_g" + path.substr(stemEnd);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <tree.json> [tree2.json ...] [--json-output <out.json>] [--threads <n>]\n";
        return 1;
    }

    vector<string> treePaths;
    string jsonOutputPath;
    int numThreads = static_cast<int>(std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--json-output") {
            if (++i < argc) jsonOutputPath = withGSuffix(argv[i]);
        } else if (arg == "--threads") {
            if (++i < argc) numThreads = std::stoi(argv[i]);
        } else {
            treePaths.push_back(arg);
        }
    }
    if (numThreads < 1) numThreads = 1;

    if (treePaths.empty()) {
        cerr << "Error: no tree files specified.\n";
        return 1;
    }

    // TORUS 56 param set
    YatfheParameters param{};
    param.n = 1024;
    param.lweNoiseB = 10;
    param.torusBase = 32;
    param.setRadixBits(14);
    param.l = 4;
    param.lApprox = 2;
    initYatfhe(param);
    auto level = param.lApprox;
    printf("n:%d, k:%d, N:%d, Torus:%d, b:%d, l:%d, lA:%d\n", param.n, param.k, param.N, param.torusBits, param.radixBits, param.l, param.lApprox);


    TlweKey tlweKey{param};
    TrgswKey trgswKey{param};
    TrlweKey& trlweKey = trgswKey.trlweKey;
    BootstrappingKeyMP bskMP{param, level};
    TlweKeySwitchingKey ksk{param};
    genTlweKey(tlweKey);
    genTrlweKey(trlweKey);
    genBootstrappingKeyMP(bskMP, trgswKey, tlweKey, param);
    TlweKey tlwe_ks_key = tlweKey;
    genTlweKeySwitchingKey(ksk, trlweKey, tlwe_ks_key, param);

    using Clock = std::chrono::steady_clock;
    using MsDouble = std::chrono::duration<double, std::milli>;
    auto measureQuiet = [](auto&& fn) -> double {
        auto t0 = Clock::now();
        fn();
        return MsDouble(Clock::now() - t0).count();
    };

    nlohmann::json results = nlohmann::json::array();

    for (const auto& path : treePaths) {
        const auto fd = loadForestFromJson(path);

        if (fd.nodes.empty() || fd.samples.empty()) {
            cerr << "Warning: no trees or samples in " << path << ", skipping.\n";
            continue;
        }

        const int nTreesTot = static_cast<int>(fd.nodes.size());
        const int nSamples = static_cast<int>(fd.samples.size());

        vector<int> initQ = fd.initQ.empty() ? vector<int>{0} : fd.initQ;
        vector<int> treeClass = fd.treeClass;
        if (treeClass.empty()) treeClass.assign(nTreesTot, 0);
        const int numClasses = static_cast<int>(initQ.size());
        const int delta = 1 << fd.kappa;

        auto argmaxClass = [&](const vector<long>& scores) {
            int best = 0;
            for (int k = 1; k < numClasses; ++k)
                if (scores[k] > scores[best]) best = k;
            return best;
        };

        cout << "\n=== " << baseName(path)
             << "  model=" << fd.model
             << "  trees=" << nTreesTot
             << "  classes=" << numClasses
             << "  kappa=" << fd.kappa
             << "  samples=" << nSamples << " ===\n";

        nlohmann::json entry;
        entry["file"] = baseName(path);
        entry["dataset"] = fd.dataset;
        entry["model"] = fd.model;
        entry["maxDepth"] = fd.maxDepth;
        entry["nTrees"] = nTreesTot;
        entry["numClasses"] = numClasses;
        entry["nFeatures"] = fd.nFeatures;
        entry["nSamples"] = nSamples;

        // --- OURS CPC (multiclass GBDT forest) ---
        {
            std::vector<DecisionTreeV2ICPC> forest;
            buildV2IForestCPC(forest, fd.nodes, fd.nFeatures, param);
            const int threads = numThreads;

            const size_t uploadBytes = ClientData{fd.nFeatures, param}.bytes();
            const size_t downloadBytes = static_cast<size_t>(numClasses) * Tlwe{param.n}.bytes;

            double sumTotal = 0;
            int correct = 0;

            for (int si = 0; si < nSamples; ++si) {
                ClientData cd{fd.nFeatures, param};
                encClientValV2I(cd, fd.samples[si], trlweKey, param);

                std::vector<Tlwe> out;
                sumTotal += measureQuiet([&]{
                    inferenceV2IGradientForestCPCTreeParallel(out, forest, treeClass, numClasses, initQ, cd,
                                                              bskMP, ksk, delta, param, threads);
                });

                // client decrypts + argmax.
                std::vector<long> scores(numClasses);
                for (int k = 0; k < numClasses; ++k)
                    scores[k] = symDecTlweToInt(out[k], tlweKey, delta);
                const int predicted = argmaxClass(scores);
                if (predicted == fd.expected[si]) correct++;
                cout << "  [OURS] sample " << si << ": pred=" << predicted
                     << "  expected=" << fd.expected[si] << "\n";
            }

            double avgTotal = sumTotal / nSamples;
            double accuracy = static_cast<double>(correct) / nSamples;
            double commUploadKB = static_cast<double>(uploadBytes) / 1024.0;
            double commDownloadKB = static_cast<double>(downloadBytes) / 1024.0;

            cout << "inferenceV2IGradientForestCPC avg: " << avgTotal << " ms\n";
            cout << "OURS accuracy: " << correct << "/" << nSamples
                 << " (" << accuracy * 100.0 << "%)\n";
            cout << "OURS comm upload: " << commUploadKB << " KB"
                 << "  download: " << commDownloadKB << " KB"
                 << "  total: " << commUploadKB + commDownloadKB << " KB\n";

            entry["OURS"] = {
                {"total_ms", avgTotal},
                {"nThreads", threads},
                {"accuracy", accuracy},
                {"commUpload_KB", commUploadKB},
                {"commDownload_KB", commDownloadKB},
                {"commTotal_KB", commUploadKB + commDownloadKB}
            };
        }

        // --- TE0 CPC (multiclass GBDT forest) ---
        {
            std::vector<DecisionTreeTE0CPC> forest;
            buildTE0ForestCPC(forest, fd.nodes, fd.nFeatures, param);
            const int threads = numThreads;

            const size_t uploadBytes = ClientDataTE0{fd.nFeatures, param}.bytes();
            const size_t downloadBytes = static_cast<size_t>(numClasses) * Tlwe{param.n}.bytes;

            double sumTotal = 0;
            int correct = 0;

            for (int si = 0; si < nSamples; ++si) {
                ClientDataTE0 cd{fd.nFeatures, param};
                encClientValTE0(cd, fd.samples[si], trlweKey, param);

                std::vector<Tlwe> out;
                sumTotal += measureQuiet([&]{
                    inferenceTE0GradientForestCPCTreeParallel(out, forest, treeClass, numClasses, initQ, cd,
                                                              bskMP, ksk, delta, param, threads);
                });

                // client decrypts + argmax.
                std::vector<long> scores(numClasses);
                for (int k = 0; k < numClasses; ++k)
                    scores[k] = symDecTlweToInt(out[k], tlweKey, delta);
                const int predicted = argmaxClass(scores);
                if (predicted == fd.expected[si]) correct++;
                cout << "  [TE0] sample " << si << ": pred=" << predicted
                     << "  expected=" << fd.expected[si] << "\n";
            }

            double avgTotal = sumTotal / nSamples;
            double accuracy = static_cast<double>(correct) / nSamples;
            double commUploadKB = static_cast<double>(uploadBytes) / 1024.0;
            double commDownloadKB = static_cast<double>(downloadBytes) / 1024.0;

            cout << "inferenceTE0GradientForestCPC avg: " << avgTotal << " ms\n";
            cout << "TE0 accuracy: " << correct << "/" << nSamples
                 << " (" << accuracy * 100.0 << "%)\n";
            cout << "TE0 comm upload: " << commUploadKB << " KB"
                 << "  download: " << commDownloadKB << " KB"
                 << "  total: " << commUploadKB + commDownloadKB << " KB\n";

            entry["TE0"] = {
                {"total_ms", avgTotal},
                {"nThreads", threads},
                {"accuracy", accuracy},
                {"commUpload_KB", commUploadKB},
                {"commDownload_KB", commDownloadKB},
                {"commTotal_KB", commUploadKB + commDownloadKB}
            };
        }

        results.push_back(std::move(entry));
    }

    nlohmann::json output;
    output["results"] = std::move(results);

    if (!jsonOutputPath.empty()) {
        std::ofstream of(jsonOutputPath);
        if (!of.is_open()) {
            cerr << "Warning: cannot write JSON output to " << jsonOutputPath << "\n";
        } else {
            of << output.dump(2) << "\n";
            cout << "\nTiming data written to: " << jsonOutputPath << "\n";
        }
    } else {
        cout << "\n--- Timing JSON (stdout) ---\n" << output.dump(2) << "\n";
    }

    return 0;
}
