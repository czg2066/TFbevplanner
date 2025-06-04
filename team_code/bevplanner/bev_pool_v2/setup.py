from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bev_pool_v2_ext',
    ext_modules=[
        CUDAExtension(
            name='bev_pool_v2_ext',
            sources=[
                'src/bev_pool.cpp', 
                'src/bev_pool_cuda.cu'  # 添加 CUDA 文件
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': ['-O3', '-Xcompiler', '-fPIC']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
