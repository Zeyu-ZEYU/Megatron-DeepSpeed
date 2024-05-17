from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="zeroc_cuda",
    ext_modules=[
        CUDAExtension(
            "zc_softmax",
            [
                "zc_softmax.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
