// Compile the production backend in this translation unit to benchmark its
// actual CUDA reduction kernel without adding a PostgreSQL entry point.
#include "../src/gftr_cuda.cu"

#include <iostream>
#include <random>

namespace {

constexpr int rows = 6'000'000;
constexpr int trials = 7;

struct Input {
    std::vector<uint64_t> keys;
    std::vector<std::vector<GftrValue>> columns;
    std::vector<int32_t> counts;
    std::vector<std::vector<GftrColumnStats>> expected;

    Input(int groups, int width)
        : keys(rows), columns(width, std::vector<GftrValue>(rows)),
          counts(groups), expected(width, std::vector<GftrColumnStats>(
              groups, GftrColumnStats{0, 0, INT_MAX, INT_MIN}))
    {
        std::mt19937 rng(178416);
        for (int row = 0; row < rows; ++row)
            keys[row] = row % groups == groups - 1 ? GFTR_NULL_KEY : row % groups;
        std::shuffle(keys.begin(), keys.end(), rng);
        for (int row = 0; row < rows; ++row) {
            int group = keys[row] == GFTR_NULL_KEY ? groups - 1 : keys[row];
            ++counts[group];
            for (int col = 0; col < width; ++col) {
                auto &value = columns[col][row];
                value.value = static_cast<int32_t>(rng() % 20001) - 10000;
                value.valid = (row + col) % 23 != 0;
                if (value.valid)
                    expected[col][group] = StatsAdd{}(expected[col][group],
                        {1, value.value, value.value, value.value});
            }
        }
    }
};

struct Timing {
    Events event;
    void begin(cudaStream_t stream)
    {
        check(cudaEventRecord(event.begin, stream), "begin timing");
    }
    void end(cudaStream_t stream)
    {
        check(cudaEventRecord(event.end, stream), "end timing");
    }
    double elapsed() const
    {
        float ms = 0;
        check(cudaEventElapsedTime(&ms, event.begin, event.end), "elapsed time");
        return ms;
    }
};

void run_case(int groups, int width)
{
    Input input(groups, width);
    Arena arena(SIZE_MAX);
    Stream owner;
    auto stream = owner.stream;
    auto keys = arena.allocate<uint64_t>(rows);
    auto sorted_keys = arena.allocate<uint64_t>(rows);
    auto unique_keys = arena.allocate<uint64_t>(rows);
    auto counts = arena.allocate<int32_t>(rows);
    auto offsets = arena.allocate<int32_t>(groups);
    auto group_count = arena.allocate<int32_t>(1);
    auto ids = arena.allocate<int32_t>(rows);
    auto sorted_ids = arena.allocate<int32_t>(rows);
    auto sorted_values = arena.allocate<GftrValue>(rows);
    std::vector<GftrValue *> values(width);
    std::vector<GftrColumnStats *> stats(width);
    for (int col = 0; col < width; ++col) {
        values[col] = arena.allocate<GftrValue>(rows);
        stats[col] = arena.allocate<GftrColumnStats>(groups);
        check(cudaMemcpyAsync(values[col], input.columns[col].data(),
              rows * sizeof(GftrValue), cudaMemcpyHostToDevice, stream), "upload payload");
    }
    check(cudaMemcpyAsync(keys, input.keys.data(), rows * sizeof(uint64_t),
          cudaMemcpyHostToDevice, stream), "upload keys");
    check(cudaStreamSynchronize(stream), "wait for input");

    size_t gftr_bytes = 0, gfur_bytes = 0, encode_bytes = 0, scan_bytes = 0;
    const auto sort_payload = [&](void *temp, int col) {
        check(cub::DeviceRadixSort::SortPairs(temp, gftr_bytes, keys,
              sorted_keys, values[col], sorted_values, rows, 0, 33, stream),
              "sort key/payload pairs");
    };
    const auto sort_ids = [&](void *temp) {
        check(cub::DeviceRadixSort::SortPairs(temp, gfur_bytes, keys,
              sorted_keys, ids, sorted_ids, rows, 0, 33, stream), "sort key/ID pairs");
    };
    const auto encode = [&](void *temp) {
        check(cub::DeviceRunLengthEncode::Encode(temp, encode_bytes, sorted_keys,
              unique_keys, counts, group_count, rows, stream), "identify groups");
    };
    const auto scan = [&](void *temp) {
        check(cub::DeviceScan::ExclusiveSum(temp, scan_bytes, counts,
              offsets, groups, stream), "compute group offsets");
    };
    sort_payload(nullptr, 0);
    sort_ids(nullptr);
    encode(nullptr);
    scan(nullptr);
    auto temp = arena.allocate<unsigned char>(
        std::max({gftr_bytes, gfur_bytes, encode_bytes, scan_bytes, size_t(1)}));
    Timing whole, discovery;
    std::vector<Timing> sort(width), reduce(width);
    std::vector<uint64_t> actual_keys(groups);
    std::vector<int32_t> actual_counts(groups);
    std::vector<GftrColumnStats> actual_stats(groups);
    const auto download = [](void *host, const void *device, size_t size) {
        check(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost), "retrieve validation data");
    };

    for (int round = 0; round <= trials; ++round)
        for (int order = 0; order < 2; ++order) {
            const bool gftr = (round + order) % 2 != 0;
            whole.begin(stream);
            sort[0].begin(stream);
            if (gftr) {
                sort_payload(temp, 0);
            } else {
                make_ids<<<std::min(65535, (rows + 255) / 256), 256, 0, stream>>>(ids, rows);
                check(cudaGetLastError(), "initialize tuple IDs");
                sort_ids(temp);
            }
            sort[0].end(stream);
            discovery.begin(stream);
            encode(temp);
            // The fixture has a known group count. Validate it after timing
            // so group discovery and all results stay on the device.
            scan(temp);
            discovery.end(stream);
            for (int col = 0; col < width; ++col) {
                if (gftr && col > 0) {
                    sort[col].begin(stream);
                    sort_payload(temp, col);
                    sort[col].end(stream);
                }
                reduce[col].begin(stream);
                int blocks = std::min(groups, 65535);
                if (gftr)
                    reduce_groups<false><<<blocks, 256, 0, stream>>>(
                        sorted_values, nullptr, offsets, counts, groups, stats[col]);
                else
                    reduce_groups<true><<<blocks, 256, 0, stream>>>(
                        values[col], sorted_ids, offsets, counts, groups, stats[col]);
                check(cudaGetLastError(), "aggregate groups");
                reduce[col].end(stream);
            }
            whole.end(stream);
            check(cudaEventSynchronize(whole.event.end), "finish aggregation");

            int actual_groups = 0;
            download(&actual_groups, group_count, sizeof(int));
            if (actual_groups != groups)
                throw std::runtime_error("unexpected group count");
            download(actual_keys.data(), unique_keys, groups * sizeof(uint64_t));
            download(actual_counts.data(), counts, groups * sizeof(int32_t));
            for (int group = 0; group < groups; ++group) {
                uint64_t key = group == groups - 1 ? GFTR_NULL_KEY : group;
                if (actual_keys[group] != key || actual_counts[group] != input.counts[group])
                    throw std::runtime_error("group keys or sizes disagree");
            }

            double sort_ms = sort[0].elapsed(), reduce_ms = 0;
            for (int col = 0; col < width; ++col) {
                if (gftr && col > 0)
                    sort_ms += sort[col].elapsed();
                reduce_ms += reduce[col].elapsed();
                download(actual_stats.data(), stats[col], groups * sizeof(GftrColumnStats));
                for (int group = 0; group < groups; ++group) {
                    const auto &a = actual_stats[group];
                    const auto &b = input.expected[col][group];
                    if (a.count != b.count || a.sum != b.sum || a.min != b.min || a.max != b.max)
                        throw std::runtime_error("aggregate statistics disagree");
                }
            }
            if (round)
                std::cout << rows << ',' << groups << ',' << width << ','
                          << (width + 1) * size_t(rows) * sizeof(uint64_t) << ','
                          << round << ',' << order << ',' << (gftr ? "GFTR" : "GFUR")
                          << ',' << sort_ms << ',' << discovery.elapsed() << ','
                          << reduce_ms << ',' << whole.elapsed() << '\n';
        }
}

} // namespace

int main()
{
    try {
        check(cudaSetDevice(0), "select CUDA device");
        std::cout << "rows,groups,width,input_bytes,round,position,mode,sort_ms,"
                     "discover_offsets_ms,reduce_ms,total_ms\n";
        for (int groups : {25, 200000})
            for (int width : {1, 4})
                run_case(groups, width);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
