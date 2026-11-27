/* TPC-H device-resident joins and a partitioned GFTR-style gather ablation. */
#include <cuda_runtime.h>
#include <cub/cub.cuh>
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
constexpr int threads = 256;

void check(cudaError_t code)
{
    if (code != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(code));
}

template<class T, bool pinned = false>
struct Buffer {
    T *data = nullptr;
    explicit Buffer(size_t n)
    {
        check(pinned ? cudaMallocHost(&data, n * sizeof(T)) : cudaMalloc(&data, n * sizeof(T)));
    }
    ~Buffer() { if (pinned) cudaFreeHost(data); else cudaFree(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};

struct Timer {
    cudaEvent_t start{}, stop{};
    Timer() { check(cudaEventCreate(&start)); check(cudaEventCreate(&stop)); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }

    template<class Fn> float ms(Fn fn)
    {
        check(cudaEventRecord(start));
        fn();
        check(cudaGetLastError());
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float elapsed;
        check(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }
};

int column(const std::string &row, size_t at)
{
    auto end = row.find('|', at);
    int value = 0;
    if (end == std::string::npos)
        throw std::runtime_error("missing TPC-H column");
    auto parsed = std::from_chars(row.data() + at, row.data() + end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != row.data() + end || value < 0)
        throw std::runtime_error("invalid TPC-H integer");
    return value;
}

template<class Fn> void read_rows(const std::string &path, Fn fn)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open " + path);
    std::string line;
    while (std::getline(input, line))
        fn(line);
    if (!input.eof())
        throw std::runtime_error("reading " + path + " failed");
}

size_t offset(const std::string &row, int field)
{
    size_t start = 0;
    for (int i = 0; i < field; ++i) {
        auto end = row.find('|', start);
        if (end == std::string::npos)
            throw std::runtime_error("missing TPC-H separator");
        start = end + 1;
    }
    return start;
}

template<class T> void upload(Buffer<T> &dst, const std::vector<T> &src)
{
    check(cudaMemcpy(dst.data, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

__global__ void scatter_lookup(const int *keys, const int *values, int n, int *map)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        map[keys[i]] = values ? values[i] : i;
}

__global__ void probe_orders(const int *keys, int n, const int *orders, int *customer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        customer[i] = orders[keys[i]];
}

struct Totals {
    long long count, customers, nations;
};

struct TotalSum {
    __host__ __device__ Totals operator()(Totals a, Totals b) const
    {
        return {a.count + b.count, a.customers + b.customers, a.nations + b.nations};
    }
};

__global__ void probe_customers(const int *customer, int n, const int *nations, Totals *rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int key = customer[i];
        int nation = key >= 0 ? nations[key] : -1;
        rows[i] = nation < 0 ? Totals{} : Totals{1, key, nation};
    }
}

std::vector<int> host_lookup(const std::vector<int> &keys, const std::vector<int> &values)
{
    std::vector<int> lookup(*std::max_element(keys.begin(), keys.end()) + 1, -1);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (lookup[keys[i]] != -1)
            throw std::runtime_error("duplicate dimension key");
        lookup[keys[i]] = values[i];
    }
    return lookup;
}

void pipeline(const std::vector<int> &fact, const std::vector<int> &orders,
              const std::vector<int> &order_customers, const std::vector<int> &customers,
              const std::vector<int> &nations)
{
    auto order_lookup = host_lookup(orders, order_customers);
    auto nation_lookup = host_lookup(customers, nations);
    Totals expected{};
    for (int order : fact) {
        int customer = order_lookup.at(order);
        if (customer >= 0) {
            int nation = nation_lookup.at(customer);
            if (nation >= 0)
                expected = TotalSum{}(expected, {1, customer, nation});
        }
    }
    std::printf("expected,%lld,%lld,%lld\n", expected.count, expected.customers, expected.nations);

    int n = static_cast<int>(fact.size());
    Timer timer;
    Buffer<int> d_fact(n), d_orders(orders.size()), d_order_customers(orders.size()),
        d_customers(customers.size()), d_nations(customers.size()),
        d_order_map(order_lookup.size()), d_nation_map(nation_lookup.size()), intermediate(n);
    Buffer<int, true> host_intermediate(n);
    Buffer<Totals> rows(n), output(1);
    size_t scratch_bytes = 0;
    check(cub::DeviceReduce::Reduce(nullptr, scratch_bytes, rows.data, output.data,
                                    n, TotalSum{}, Totals{}));
    Buffer<unsigned char> scratch(scratch_bytes);
    auto load = [&] {
        upload(d_fact, fact);
        upload(d_orders, orders);
        upload(d_order_customers, order_customers);
        upload(d_customers, customers);
        upload(d_nations, nations);
        check(cudaMemset(d_order_map.data, 0xff, order_lookup.size() * sizeof(int)));
        check(cudaMemset(d_nation_map.data, 0xff, nation_lookup.size() * sizeof(int)));
        scatter_lookup<<<(orders.size() + threads - 1) / threads, threads>>>(
            d_orders.data, d_order_customers.data, orders.size(), d_order_map.data);
        scatter_lookup<<<(customers.size() + threads - 1) / threads, threads>>>(
            d_customers.data, d_nations.data, customers.size(), d_nation_map.data);
    };
    timer.ms(load);

    std::array<std::string, 3> modes{"cold_upload", "resident", "intermediate_roundtrip"};
    std::mt19937 random(914);
    for (int trial = 0; trial <= 7; ++trial) {
        std::shuffle(modes.begin(), modes.end(), random);
        for (const auto &mode : modes) {
            auto start = Clock::now();
            float loading = mode == "cold_upload" ? timer.ms(load) : 0;
            float first = timer.ms([&] {
                probe_orders<<<(n + threads - 1) / threads, threads>>>(
                    d_fact.data, n, d_order_map.data, intermediate.data);
            });
            float transfers = mode == "intermediate_roundtrip" ? timer.ms([&] {
                check(cudaMemcpy(host_intermediate.data,
                                 intermediate.data, n * sizeof(int), cudaMemcpyDeviceToHost));
                check(cudaMemcpy(intermediate.data, host_intermediate.data, n * sizeof(int),
                                 cudaMemcpyHostToDevice));
            }) : 0;
            float second = timer.ms([&] {
                probe_customers<<<(n + threads - 1) / threads, threads>>>(
                    intermediate.data, n, d_nation_map.data, rows.data);
                check(cub::DeviceReduce::Reduce(scratch.data, scratch_bytes, rows.data,
                                                output.data, n, TotalSum{}, Totals{}));
            });
            Totals actual{};
            check(cudaMemcpy(&actual, output.data, sizeof(actual), cudaMemcpyDeviceToHost));
            double wall = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            if (actual.count != expected.count || actual.customers != expected.customers ||
                actual.nations != expected.nations)
                throw std::runtime_error("incorrect resident pipeline totals");
            if (trial)
                std::printf("pipeline,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                            mode.c_str(), trial, n, loading, first, transfers, second, wall);
        }
    }
}

__host__ __device__ unsigned payload(int key, int size, int col)
{
    return col == 0 ? static_cast<unsigned>(size) :
        (static_cast<unsigned>(key) * 2654435761u + 97u * col) & 65535u;
}

__global__ void generate_payloads(const int *keys, const int *sizes, int n, int width, unsigned *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        for (int col = 0; col < width; ++col)
            out[col * n + i] = payload(keys[i], sizes[i], col);
}

__global__ void transform_payloads(const unsigned *input, const int *order, int n,
                                  int width, unsigned *output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        for (int col = 0; col < width; ++col)
            output[col * n + i] = input[col * n + order[i]];
}

__global__ void initialize_rows(int *rows, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        rows[i] = i;
}

__global__ void gather_matches(const int *probe, int n, const int *lookup,
                               const int *original_rows, const unsigned *payloads,
                               int build_rows, int width, unsigned *output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int row = lookup[probe[i]];
        if (original_rows)
            row = original_rows[row];
        for (int col = 0; col < width; ++col)
            output[(size_t) col * n + i] = payloads[col * build_rows + row];
    }
}

void gather_ablation(const std::vector<int> &fact, const std::vector<int> &parts,
                     const std::vector<int> &sizes, int width)
{
    auto lookup = host_lookup(parts, sizes);
    if (std::any_of(fact.begin(), fact.end(), [&](int key) { return lookup.at(key) < 0; }))
        throw std::runtime_error("part lookup requires matched TPC-H keys");
    int n = static_cast<int>(fact.size());
    int m = static_cast<int>(parts.size());
    int fact_blocks = (n + threads - 1) / threads;
    int part_blocks = (m + threads - 1) / threads;
    constexpr int partition_bits = 8;
    Timer timer;
    Buffer<int> d_fact(n), sorted_probe(n), d_parts(m), d_sizes(m),
        sorted_parts(m), row_ids(m), sorted_ids(m), original_map(lookup.size()),
        partitioned_map(lookup.size());
    Buffer<unsigned> original_payload((size_t) m * width), transformed((size_t) m * width),
        output((size_t) n * width);
    upload(d_fact, fact);
    upload(d_parts, parts);
    upload(d_sizes, sizes);
    generate_payloads<<<part_blocks, threads>>>(d_parts.data, d_sizes.data, m, width,
                                               original_payload.data);
    initialize_rows<<<part_blocks, threads>>>(row_ids.data, m);
    scatter_lookup<<<part_blocks, threads>>>(d_parts.data, row_ids.data, m, original_map.data);
    size_t build_bytes = 0, probe_bytes = 0;
    check(cub::DeviceRadixSort::SortPairs(nullptr, build_bytes, d_parts.data, sorted_parts.data,
                                         row_ids.data, sorted_ids.data, m, 0, partition_bits));
    check(cub::DeviceRadixSort::SortKeys(nullptr, probe_bytes, d_fact.data, sorted_probe.data,
                                        n, 0, partition_bits));
    size_t cap = std::max(build_bytes, probe_bytes);
    Buffer<unsigned char> temp(cap);
    std::array<std::string, 3> modes{"direct", "partitioned_gfur", "partitioned_gftr"};
    std::mt19937 random(117);
    std::vector<int> partition_keys(n);
    std::vector<unsigned> column_output(n);

    for (int trial = 0; trial <= 7; ++trial) {
        std::shuffle(modes.begin(), modes.end(), random);
        for (const auto &mode : modes) {
            bool partitioned = mode != "direct";
            bool gftr = mode == "partitioned_gftr";
            auto start = Clock::now();
            float partition = partitioned ? timer.ms([&] {
                check(cub::DeviceRadixSort::SortPairs(temp.data, cap, d_parts.data, sorted_parts.data,
                                                     row_ids.data, sorted_ids.data, m, 0, partition_bits));
                check(cub::DeviceRadixSort::SortKeys(temp.data, cap, d_fact.data, sorted_probe.data,
                                                    n, 0, partition_bits));
                scatter_lookup<<<part_blocks, threads>>>(sorted_parts.data, nullptr, m, partitioned_map.data);
            }) : 0;
            float transform = gftr ? timer.ms([&] {
                transform_payloads<<<part_blocks, threads>>>(original_payload.data, sorted_ids.data,
                                                             m, width, transformed.data);
            }) : 0;
            float gather = timer.ms([&] {
                gather_matches<<<fact_blocks, threads>>>(
                    partitioned ? sorted_probe.data : d_fact.data, n,
                    partitioned ? partitioned_map.data : original_map.data,
                    partitioned && !gftr ? sorted_ids.data : nullptr,
                    gftr ? transformed.data : original_payload.data, m, width, output.data);
            });
            double wall = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

            // Exact column checks outside the timed cold or warm measurements.
            if (trial == 0) {
                if (partitioned)
                    check(cudaMemcpy(partition_keys.data(), sorted_probe.data, n * sizeof(int),
                                     cudaMemcpyDeviceToHost));
                const auto &keys = partitioned ? partition_keys : fact;
                for (int col = 0; col < width; ++col) {
                    check(cudaMemcpy(column_output.data(), output.data + (size_t) col * n,
                                     n * sizeof(unsigned), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < n; ++i)
                        if (column_output[i] != payload(keys[i], lookup[keys[i]], col))
                            throw std::runtime_error("incorrect GFTR or GFUR payload");
                }
            }
            if (trial)
                std::printf("gather,%s,%d,%d,%.4f,%.4f,%.4f,%.4f\n",
                            mode.c_str(), trial, width, partition, transform, gather, wall);
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
        std::string base = argv[1];
        std::vector<int> fact_orders, fact_parts, orders, order_customers, customers, nations, parts, sizes;
        auto start = Clock::now();
        read_rows(base + "/lineitem.tbl", [&](const auto &row) {
            fact_orders.push_back(column(row, 0));
            fact_parts.push_back(column(row, offset(row, 1)));
        });
        read_rows(base + "/orders.tbl", [&](const auto &row) {
            orders.push_back(column(row, 0));
            order_customers.push_back(column(row, offset(row, 1)));
        });
        read_rows(base + "/customer.tbl", [&](const auto &row) {
            customers.push_back(column(row, 0));
            nations.push_back(column(row, offset(row, 3)));
        });
        read_rows(base + "/part.tbl", [&](const auto &row) {
            parts.push_back(column(row, 0));
            sizes.push_back(column(row, offset(row, 5)));
        });
        if (fact_orders.empty() || orders.empty() || customers.empty() || parts.empty() ||
            fact_orders.size() >= INT32_MAX)
            throw std::runtime_error("unexpected TPC-H input cardinality");
        std::fprintf(stderr, "Projected TPC-H input in %.3f seconds\n",
                     std::chrono::duration<double>(Clock::now() - start).count());
        pipeline(fact_orders, orders, order_customers, customers, nations);
        for (int width : {1, 8})
            gather_ablation(fact_parts, parts, sizes, width);
    } catch (const std::exception &err) {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }
}
