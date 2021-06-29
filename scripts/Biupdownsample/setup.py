from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_ARGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]

setup(
    name='biupdownsample',
    ext_modules=[
        CUDAExtension(
            'biupsample_naive_cuda', [
                'src/cuda/biupsample_naive_cuda.cpp', 'src/cuda/biupsample_naive_cuda_kernel.cu',
                'src/biupsample_naive_ext.cpp'
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            }),
        CUDAExtension(
            'bidownsample_naive_cuda', [
                'src/cuda/bidownsample_naive_cuda.cpp', 'src/cuda/bidownsample_naive_cuda_kernel.cu',
                'src/bidownsample_naive_ext.cpp'
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            })
    ],
    cmdclass={'build_ext': BuildExtension})
