/* Standalone CUDA join probe on TPC-H orders and lineitem .tbl projections. */
#include "gftr_join_lookup.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

void check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

template<class T, bool Pinned = false> struct Buffer {
    T *data = nullptr;

    explicit Buffer(size_t n)
    {
        check(Pinned ? cudaMallocHost(&data, n * sizeof(T)) :
                       cudaMalloc(&data, n * sizeof(T)));
    }
    ~Buffer() { if (Pinned) cudaFreeHost(data); else cudaFree(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};

struct Timing {
    cudaStream_t stream{};
    cudaEvent_t start{}, stop{};

    Timing()
    {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&stop));
    }
    ~Timing()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
    }
    template<class Fn> float measure(Fn fn)
    {
        check(cudaEventRecord(start, stream));
        fn();
        check(cudaEventRecord(stop, stream));
        check(cudaEventSynchronize(stop));
        float ms;
        check(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

struct Match {
    int probe;
    int build;
};

__global__ void probe_keys(const void *table, GftrJoinLayout layout,
                           const int *keys, int n, int *dense,
                           Match *compact, unsigned *count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int match = -1;
    if (i < n) {
        int key = keys[i];
        match = layout.dense ?
            gftr_join_dense_lookup(static_cast<const int *>(table), layout, key) :
            gftr_join_hash_lookup(static_cast<const GftrJoinEntry *>(table), layout.slots, key);
    }
    if (!compact) {
        if (i < n)
            dense[i] = match;
        return;
    }
    unsigned mask = __ballot_sync(0xffffffff, match >= 0);
    if (mask == 0)
        return;
    unsigned base = 0;
    unsigned lane = threadIdx.x % 32;
    if (lane == 0)
        base = atomicAdd(count, __popc(mask));
    base = __shfl_sync(0xffffffff, base, 0);
    if (match >= 0)
        compact[base + __popc(mask & ((1u << lane) - 1))] = Match{i, match};
}

struct Input {
    std::vector<int> keys;
    std::vector<int> filtered;
};

Input read_keys(const std::string &path, bool with_date)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open " + path);
    Input result;
    std::string line;
    while (std::getline(file, line)) {
        auto separator = line.find('|');
        if (separator == std::string::npos)
            throw std::runtime_error("missing key delimiter");
        int key;
        auto end = line.data() + separator;
        auto parsed = std::from_chars(line.data(), end, key);
        if (parsed.ec != std::errc{} || parsed.ptr != end || key <= 0)
            throw std::runtime_error("invalid key");
        result.keys.push_back(key);
        if (with_date) {
            for (int i = 0; i < 3; i++) {
                separator = line.find('|', separator + 1);
                if (separator == std::string::npos)
                    throw std::runtime_error("missing order date");
            }
            if (line.compare(separator + 1, 8, "1995-01-") == 0)
                result.filtered.push_back(key);
        }
    }
    if (!file.eof() || result.keys.empty() || result.keys.size() >= INT32_MAX)
        throw std::runtime_error("invalid input row count or read failure");
    return result;
}

void run(const char *name, const std::vector<int> &build, const std::vector<int> &probe)
{
    if (build.empty())
        throw std::runtime_error("empty build input");
    GftrJoinLayout layout{};
    if (gftr_join_layout(build.data(), build.size(), &layout))
        throw std::runtime_error("lookup size overflow");
    std::vector<int> dense;
    std::vector<GftrJoinEntry> hash;
    const void *host_table;
    if (layout.dense) {
        dense.assign(layout.slots, -1);
        for (size_t i = 0; i < build.size(); ++i) {
            int &slot = dense[build[i] - layout.minimum];
            if (slot != -1)
                throw std::runtime_error("duplicate build key");
            slot = i;
        }
        host_table = dense.data();
    } else {
        hash.assign(layout.slots, GFTR_JOIN_EMPTY);
        for (size_t i = 0; i < build.size(); ++i) {
            size_t slot = gftr_join_hash(build[i], layout.slots - 1);
            while (hash[slot] != GFTR_JOIN_EMPTY) {
                if (gftr_join_is_key(hash[slot], build[i]))
                    throw std::runtime_error("duplicate build key");
                slot = (slot + 1) & (layout.slots - 1);
            }
            hash[slot] = gftr_join_entry(build[i], i);
        }
        host_table = hash.data();
    }

    std::vector<int> expected;
    expected.reserve(probe.size());
    unsigned hits = 0;
    for (int key : probe) {
        int match = layout.dense ?
            gftr_join_dense_lookup(dense.data(), layout, key) :
            gftr_join_hash_lookup(hash.data(), layout.slots, key);
        expected.push_back(match);
        hits += match >= 0;
    }

    int n = static_cast<int>(probe.size());
    Timing timing;
    Buffer<unsigned char> table(layout.bytes);
    Buffer<int> keys(n), matches(n);
    Buffer<Match> pairs(n);
    Buffer<unsigned> count(1);
    Buffer<int, true> host_keys(n), host_matches(n);
    Buffer<Match, true> host_pairs(n);
    Buffer<unsigned, true> host_count(1);
    std::copy(probe.begin(), probe.end(), host_keys.data);
    float build_upload = timing.measure([&] {
        check(cudaMemcpyAsync(table.data, host_table, layout.bytes,
                              cudaMemcpyHostToDevice, timing.stream));
    });
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0));
    size_t reserved = std::min<size_t>(properties.persistingL2CacheMaxSize, layout.bytes);
    size_t window = std::min<size_t>(properties.accessPolicyMaxWindowSize, layout.bytes);
    if (reserved && window)
        check(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserved));

    std::array<std::string, 4> modes{"resident", "resident_l2", "dense_return", "compact_return"};
    std::mt19937 random(714);
    std::vector<int> seen(n);
    for (int trial = 0; trial <= 7; ++trial) {
        std::shuffle(modes.begin(), modes.end(), random);
        for (const auto &mode : modes) {
            bool resident = mode == "resident" || mode == "resident_l2";
            bool compact = mode == "compact_return";
            if (mode == "resident_l2" && (!reserved || !window))
                continue;
            cudaStreamAttrValue policy{};
            if (mode == "resident_l2") {
                policy.accessPolicyWindow.base_ptr = table.data;
                policy.accessPolicyWindow.num_bytes = window;
                policy.accessPolicyWindow.hitRatio = std::min(1.0f, float(reserved) / window);
                policy.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                policy.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
            }
            check(cudaStreamSetAttribute(timing.stream, cudaStreamAttributeAccessPolicyWindow, &policy));
            check(cudaCtxResetPersistingL2Cache());

            auto upload = [&] {
                check(cudaMemcpyAsync(keys.data, host_keys.data, n * sizeof(int),
                                      cudaMemcpyHostToDevice, timing.stream));
            };
            auto launch = [&] {
                if (compact)
                    check(cudaMemsetAsync(count.data, 0, sizeof(unsigned), timing.stream));
                probe_keys<<<(n + 255) / 256, 256, 0, timing.stream>>>(
                    table.data, layout, keys.data, n, matches.data,
                    compact ? pairs.data : nullptr, count.data);
                check(cudaGetLastError());
            };
            if (resident) {
                // Both cache policies get one untimed warm probe on resident keys.
                upload();
                launch();
                check(cudaStreamSynchronize(timing.stream));
            }
            auto start = Clock::now();
            float h2d = resident ? 0 : timing.measure(upload);
            float kernel = timing.measure(launch);
            float d2h = 0;
            size_t output_bytes = resident ? 0 : size_t(n) * sizeof(int);
            if (compact) {
                d2h = timing.measure([&] {
                    check(cudaMemcpyAsync(host_count.data, count.data, sizeof(unsigned),
                                          cudaMemcpyDeviceToHost, timing.stream));
                });
                if (*host_count.data != hits)
                    throw std::runtime_error("incorrect compact output count");
                output_bytes = sizeof(unsigned) + size_t(hits) * sizeof(Match);
                if (hits)
                    d2h += timing.measure([&] {
                        check(cudaMemcpyAsync(host_pairs.data, pairs.data, hits * sizeof(Match),
                                              cudaMemcpyDeviceToHost, timing.stream));
                    });
            } else if (!resident) {
                d2h = timing.measure([&] {
                    check(cudaMemcpyAsync(host_matches.data, matches.data, output_bytes,
                                          cudaMemcpyDeviceToHost, timing.stream));
                });
            }
            double wall = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

            if (compact) {
                std::fill(seen.begin(), seen.end(), -1);
                for (unsigned i = 0; i < hits; ++i) {
                    Match pair = host_pairs.data[i];
                    if (pair.probe < 0 || pair.probe >= n || seen[pair.probe] != -1)
                        throw std::runtime_error("invalid compact match index");
                    seen[pair.probe] = pair.build;
                }
                if (seen != expected)
                    throw std::runtime_error("incorrect compact output");
            } else {
                // Read resident results after stopping the timer to verify them.
                if (resident)
                    check(cudaMemcpy(host_matches.data, matches.data, n * sizeof(int),
                                     cudaMemcpyDeviceToHost));
                if (!std::equal(expected.begin(), expected.end(), host_matches.data))
                    throw std::runtime_error("incorrect dense output");
            }
            if (trial)
                std::printf("%s,%s,%s,%d,%zu,%d,%u,%zu,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                            name, layout.dense ? "dense" : "hash", mode.c_str(),
                            trial, layout.bytes, n, hits, output_bytes, build_upload,
                            h2d, kernel, d2h, wall);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s TPC_H_DATA_DIRECTORY\n", argv[0]);
        return 2;
    }
    try {
        check(cudaSetDevice(0));
        std::string directory(argv[1]);
        auto start = Clock::now();
        Input orders = read_keys(directory + "/orders.tbl", true);
        Input lines = read_keys(directory + "/lineitem.tbl", false);
        std::fprintf(stderr, "Loaded projections in %.3f s (untimed)\n",
                     std::chrono::duration<double>(Clock::now() - start).count());
        std::puts("case,lookup,mode,trial,lookup_bytes,probe_rows,matches,output_bytes,"
                  "lookup_upload_ms,h2d_ms,kernel_ms,d2h_ms,wall_ms");
        run("orders_broad", orders.keys, lines.keys);
        run("orders_january", orders.filtered, lines.keys);
    } catch (const std::exception &err) {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }
}
