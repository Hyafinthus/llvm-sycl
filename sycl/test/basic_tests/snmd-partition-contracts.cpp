// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue;
  sycl::buffer<float, 3> input{sycl::range<3>{8, 8, 8}};
  sycl::buffer<float, 3> output{sycl::range<3>{8, 8, 8}};
  queue.submit([&](sycl::handler &cgh) {
    auto read = input.get_access<sycl::access::mode::read>(cgh);
    auto write = output.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.ext_snmd_partition_halo(read, 1, 1);
    cgh.ext_snmd_partition_local(write);
    cgh.parallel_for(sycl::range<3>{8, 8, 8}, [=](sycl::id<3> id) {
      const std::size_t before = id[0] == 0 ? 0 : id[0] - 1;
      const std::size_t after = id[0] == 7 ? 7 : id[0] + 1;
      write[id] = read[{before, id[1], id[2]}] + read[id] +
                  read[{after, id[1], id[2]}];
    });
  });
}
