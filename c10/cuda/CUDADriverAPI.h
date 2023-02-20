#pragma once
#ifndef C10_MOBILE

#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>
#include <cuda.h>

namespace c10 {
namespace cuda {
class C10_CUDA_API CUDADriverAPI {
 public:
  CUDADriverAPI();
  ~CUDADriverAPI();
  bool hasPrimaryContext(int device);

 private:
  typedef CUresult (*_cuDevicePrimaryCtxGetState)(
      CUdevice dev,
      unsigned int* flags,
      int* active);
  bool is_api_initialized;
  void* handle;
  _cuDevicePrimaryCtxGetState _hasPrimaryContext_funcptr;
  void initialize_api();
  void destroy_handle();
};
} // namespace cuda
} // namespace c10
#endif // C10_MOBILE
