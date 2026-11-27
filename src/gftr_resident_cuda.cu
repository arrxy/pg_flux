#include "gftr_resident.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace {
constexpr int threads = 256;

void check(cudaError_t code)
{
    if (code != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(code));
}

int blocks(int32_t rows)
{
    return (static_cast<unsigned>(rows) + threads - 1) / threads;
}

template<class T>
struct DeviceBuffer {
    T *ptr = nullptr;

    void allocate(size_t n)
    {
        check(cudaMalloc(&ptr, n * sizeof(T)));
    }

    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct Totals {
    int64_t count, customers, nations;
};

struct Combine {
    __host__ __device__ Totals operator()(Totals a, Totals b) const
    {
        return {a.count + b.count, a.customers + b.customers, a.nations + b.nations};
    }
};

__global__ void build_map(const int32_t *keys, const int32_t *values,
                          int count, int32_t *map, int *duplicate)
{
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < static_cast<unsigned>(count) &&
        atomicCAS(map + keys[row], -1, values[row]) != -1)
        atomicExch(duplicate, 1);
}

__global__ void probe_orders(const int32_t *facts, int n, const int32_t *orders,
                             size_t slots, int32_t *output)
{
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < static_cast<unsigned>(n)) {
        unsigned key = facts[row];
        output[row] = key < slots ? orders[key] : -1;
    }
}

struct ProbeCustomers {
    const int32_t *nations;
    size_t slots;

    __device__ Totals operator()(int32_t customer) const
    {
        unsigned key = customer;
        int32_t nation = key < slots ? nations[key] : -1;
        return nation < 0 ? Totals{} : Totals{1, customer, nation};
    }
};

int32_t max_nonnegative(const int32_t *array, int32_t count)
{
    int32_t maximum = 0;
    for (int32_t i = 0; i < count; i++) {
        if (array[i] < 0)
            throw std::runtime_error("dimension columns require nonnegative int4 values");
        maximum = std::max(maximum, array[i]);
    }
    return maximum;
}
}

struct GftrResident {
    int device;
    int n;
    size_t order_slots, customer_slots;
    DeviceBuffer<int32_t> facts, orders, customers, intermediate;
    DeviceBuffer<Totals> result;
    DeviceBuffer<unsigned char> scratch;
    size_t scratch_bytes = 0;

    explicit GftrResident(int device) : device(device) {}
    ~GftrResident() { cudaSetDevice(device); }

    auto customer_probe()
    {
        return cub::TransformInputIterator<Totals, ProbeCustomers, const int32_t *>(
            intermediate.ptr, {customers.ptr, customer_slots});
    }

    void initialize(const GftrResidentInput &in, size_t budget)
    {
        n = in.fact_count;
        order_slots = size_t(max_nonnegative(in.orders, in.order_count)) + 1;
        customer_slots = size_t(max_nonnegative(in.customers, in.customer_count)) + 1;
        max_nonnegative(in.order_customers, in.order_count);
        max_nonnegative(in.nations, in.customer_count);
        if (n == 0 || in.order_count == 0 || in.customer_count == 0) {
            n = 0;
            return;
        }
        check(cub::DeviceReduce::Reduce(nullptr, scratch_bytes, customer_probe(),
                                        result.ptr, n, Combine{}, Totals{}));
        size_t dimension_rows = std::max(in.order_count, in.customer_count);
        size_t map_scratch = 2 * dimension_rows * sizeof(int32_t) + sizeof(int);
        size_t bytes = (order_slots + customer_slots + 2 * size_t(n)) * sizeof(int32_t)
            + sizeof(Totals) + std::max(scratch_bytes, map_scratch);
        size_t free_bytes, total_bytes;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (bytes > std::min(budget, free_bytes - free_bytes / 10))
            throw std::runtime_error("resident join exceeds the device memory budget");
        facts.allocate(n);
        orders.allocate(order_slots);
        customers.allocate(customer_slots);
        intermediate.allocate(n);
        result.allocate(1);
        check(cudaMemcpy(facts.ptr, in.facts, size_t(n) * sizeof(int32_t), cudaMemcpyHostToDevice));
        {
            DeviceBuffer<int32_t> keys, values;
            DeviceBuffer<int> duplicate;
            keys.allocate(dimension_rows);
            values.allocate(dimension_rows);
            duplicate.allocate(1);
            auto load_map = [&](const int32_t *host_keys, const int32_t *host_values,
                                int count, int32_t *map, size_t slots) {
                size_t bytes = size_t(count) * sizeof(int32_t);
                check(cudaMemcpy(keys.ptr, host_keys, bytes, cudaMemcpyHostToDevice));
                check(cudaMemcpy(values.ptr, host_values, bytes, cudaMemcpyHostToDevice));
                check(cudaMemset(map, 0xff, slots * sizeof(int32_t)));
                check(cudaMemset(duplicate.ptr, 0, sizeof(int)));
                build_map<<<blocks(count), threads>>>(keys.ptr, values.ptr, count, map, duplicate.ptr);
                check(cudaGetLastError());
                int collision;
                check(cudaMemcpy(&collision, duplicate.ptr, sizeof(int), cudaMemcpyDeviceToHost));
                if (collision)
                    throw std::runtime_error("resident join requires unique dimension keys");
            };
            load_map(in.orders, in.order_customers, in.order_count, orders.ptr, order_slots);
            load_map(in.customers, in.nations, in.customer_count, customers.ptr, customer_slots);
        }
        scratch.allocate(std::max(size_t(1), scratch_bytes));
    }
};

extern "C" void gftr_resident_free(GftrResident *state)
{
    delete state;
}

extern "C" int gftr_resident_load(const GftrResidentInput *input, int device, size_t budget,
                                   GftrResident **out, char error[GFTR_ERROR_SIZE])
{
    *out = nullptr;
    try {
        check(cudaSetDevice(device));
        auto state = std::make_unique<GftrResident>(device);
        state->initialize(*input, budget);
        *out = state.release();
        return 0;
    } catch (const std::exception &exception) {
        std::snprintf(error, GFTR_ERROR_SIZE, "%s", exception.what());
        return -1;
    }
}

extern "C" int gftr_resident_run(GftrResident *state, int64_t out[3], char error[GFTR_ERROR_SIZE])
{
    try {
        if (state->n == 0) {
            std::fill_n(out, 3, 0);
            return 0;
        }
        check(cudaSetDevice(state->device));
        probe_orders<<<blocks(state->n), threads>>>(state->facts.ptr, state->n, state->orders.ptr,
                                                     state->order_slots, state->intermediate.ptr);
        check(cudaGetLastError());
        check(cub::DeviceReduce::Reduce(state->scratch.ptr, state->scratch_bytes,
                                        state->customer_probe(), state->result.ptr,
                                        state->n, Combine{}, Totals{}));
        Totals totals;
        check(cudaMemcpy(&totals, state->result.ptr, sizeof(totals), cudaMemcpyDeviceToHost));
        out[0] = totals.count;
        out[1] = totals.customers;
        out[2] = totals.nations;
        return 0;
    } catch (const std::exception &exception) {
        std::snprintf(error, GFTR_ERROR_SIZE, "%s", exception.what());
        return -1;
    }
}
