//
// Created by xsong93 on 6/16/26.
//
//
#include "include/breakdown_common.h"
#include "include/thetaEq0_decision_tree.h"

static constexpr const char* SCHEME = "TE0";

static void setSchemeParams(YatfheParameters& param) {
    param.n = 760;
    param.lweNoiseB = 16;
    param.torusBase = 16;
    param.setRadixBits(7);
    param.l = 4;
    param.lApprox = 3;
}

int main(int argc, char* argv[]) {
    BenchOptions opt;
    if (!parseArgs(argc, argv, SCHEME, opt)) return 1;

    YatfheParameters param{};
    setSchemeParams(param);
    applyOverrides(param, opt.overrides);
    initYatfhe(param);
    const auto level = param.lApprox;
    printParams(param);

    EvalKeys keys{param, level};
    auto& trlweKey = keys.trlweKey();

    const SetupCost setup = measureSetupCost(keys.bskMP, keys.ksk, param);
    printSetupCost(setup, keys.bskMP, keys.ksk, param, level);

    nlohmann::json results = nlohmann::json::array();

    for (const auto& path : opt.treePaths) {
        const auto fd = loadForestFromJson(path);

        if (fd.nodes.empty() || fd.samples.empty()) {
            cerr << "Warning: no trees or samples in " << path << ", skipping.\n";
            continue;
        }

        const auto& treeNodes = fd.nodes[0];
        const int nodeCount = static_cast<int>(treeNodes.size());
        const int leafCount = computeLeafCount(treeNodes);
        const int nSamples = static_cast<int>(fd.samples.size());

        cout << "\n=== " << baseName(path)
             << "  depth=" << fd.maxDepth
             << "  nodes=" << nodeCount
             << "  leaves=" << leafCount
             << "  samples=" << nSamples << " ===\n";

        nlohmann::json entry;
        entry["file"] = baseName(path);
        entry["dataset"] = fd.dataset;
        entry["maxDepth"] = fd.maxDepth;
        entry["nTrees"] = fd.nTrees;
        entry["nFeatures"] = fd.nFeatures;
        entry["nSamples"] = nSamples;

        DecisionTreeTE0CPC tree{nodeCount, leafCount, fd.nFeatures, param};
        buildTE0TreeCPC(tree, treeNodes, param);
        const int threads = opt.numThreads;

        const size_t uploadBytes = ClientDataTE0{tree.featureCount, param}.bytes();
        const size_t downloadBytes = Tlwe{param.n}.bytes;

        std::optional<DecisionTreeTE0CPCHelper> warmHelper;
        if (opt.warmupRotations > 0) {
            warmHelper.emplace(tree, param);
            ClientDataTE0 warmCd{tree.featureCount, param};
            encClientValTE0(warmCd, fd.samples[0], trlweKey, param);
            rotTE0TreeCPC(*warmHelper, tree, warmCd, param, threads);
            sumTE0TreePathCPC(*warmHelper, tree, keys.ksk, param, threads);
        }

        const int warmCalls = warmHelper
            ? (opt.warmupRotations + static_cast<int>(warmHelper->pathSum.size()) - 1) / static_cast<int>(warmHelper->pathSum.size())
            : 0;

        double sumRot = 0, sumSum = 0, sumCal = 0, sumGroup = 0;
        double cpuRot = 0, cpuSum = 0, cpuCal = 0, cpuGroup = 0;
        int correct = 0;

        for (int si = 0; si < nSamples; ++si) {
            DecisionTreeTE0CPCHelper treeHelper{tree, param};
            ClientDataTE0 cd{tree.featureCount, param};
            encClientValTE0(cd, fd.samples[si], trlweKey, param);

            StageTime st;
            st = measureBoth([&]{ rotTE0TreeCPC(treeHelper, tree, cd, param, threads); });
            sumRot += st.wallMs;
            cpuRot += st.cpuMs;
            st = measureBoth([&]{ sumTE0TreePathCPC(treeHelper, tree, keys.ksk, param, threads); });
            sumSum += st.wallMs;
            cpuSum += st.cpuMs;

            for (int w = 0; w < warmCalls; ++w) {
                calTE0TreeCPC(*warmHelper, keys.bskMP, param, threads);
            }

            st = measureBoth([&]{ calTE0TreeCPC(treeHelper, keys.bskMP, param, threads); });
            sumCal += st.wallMs;
            cpuCal += st.cpuMs;

            Tlwe res{param.n};
            st = measureBoth([&]{ groupLeafTE0TreeCPC(res, treeHelper, keys.ksk, param); });
            sumGroup += st.wallMs;
            cpuGroup += st.cpuMs;

            auto r = symDecTlweToInt(res, keys.tlweKey, param.torusBase);
            if (r == fd.expected[si]) {correct++;}
            cout << "  [TE0] sample " << si << ": res=" << r << "  expected=" << fd.expected[si] << "\n";
        }

        double avgRot = sumRot / nSamples;
        double avgSum = sumSum / nSamples;
        double avgCal = sumCal / nSamples;
        double avgGroup = sumGroup / nSamples;
        double accuracy = static_cast<double>(correct) / nSamples;
        double commUploadKB = static_cast<double>(uploadBytes) / 1024.0;
        double commDownloadKB = static_cast<double>(downloadBytes) / 1024.0;

        cout << "rotTE0TreeCPC avg: " << avgRot << " ms\n";
        cout << "sumTE0TreePathCPC avg: " << avgSum << " ms\n";
        cout << "calTE0TreeCPC avg: " << avgCal  << " ms\n";
        cout << "groupLeafTE0TreeCPC avg: " << avgGroup << " ms\n";
        cout << "TE0 warmup: " << opt.warmupRotations << " untimed rotations per selectLeaf call\n";
        cout << "TE0 accuracy: " << correct << "/" << nSamples << " (" << accuracy * 100.0 << "%)\n";
        cout << "TE0 comm upload: " << commUploadKB << " KB"
             << "  download: " << commDownloadKB << " KB"
             << "  total: " << commUploadKB + commDownloadKB << " KB\n";

        entry[SCHEME] = {
            {"nThreads", threads},
            {"warmup", opt.warmupRotations},
            {"compareTree_ms", avgRot},
            {"sumPath_ms", avgSum},
            {"selectLeaf_ms", avgCal},
            {"groupLeaf_ms", avgGroup},
            {"total_ms", avgRot + avgSum + avgCal + avgGroup},
            {"compareTree_cpu_ms", cpuRot / nSamples},
            {"sumPath_cpu_ms", cpuSum / nSamples},
            {"selectLeaf_cpu_ms", cpuCal / nSamples},
            {"groupLeaf_cpu_ms", cpuGroup / nSamples},
            {"total_cpu_ms", (cpuRot + cpuSum + cpuCal + cpuGroup) / nSamples},
            {"accuracy", accuracy},
            {"commUpload_KB", commUploadKB},
            {"commDownload_KB", commDownloadKB},
            {"commTotal_KB", commUploadKB + commDownloadKB},
            {"setupUpload_MB", setup.totalMB()}
        };
        results.push_back(std::move(entry));
    }

    nlohmann::json output;
    output["scheme"] = SCHEME;
    output["setup"] = setupToJson(setup, keys.bskMP, keys.ksk, param, level);
    output["results"] = std::move(results);
    writeOutput(output, opt.jsonOutputPath);

    return 0;
}
