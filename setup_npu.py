# Copyright (c) 2023, Tri Dao.

import sys
import functools
import warnings
import os
import re
import ast
import glob
import shutil
import logging
import sysconfig
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension
from torch_npu.testing.common_utils import set_npu_device


import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
    IS_HIP_EXTENSION,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# class CMakeExtension(Extension):
#     def __init__(self, name, sourcedir=""):
#         super().__init__(name, sources=[])
#         self.sourcedir = os.path.abspath(sourcedir)


# class CMakeBuild(build_ext):
#     def run(self):
#         for ext in self.extensions:
#             self.build_cmake(ext)
#             self.generate_pyi(ext)

#     def build_cmake(self, ext):
#         extdir = os.path.abspath(os.path.dirname(
#             self.get_ext_fullpath(ext.name)))
#         # 确保输出目录存在
#         os.makedirs(extdir, exist_ok=True)

#         logging.info(f"CMakeLists.txt所在目录：{ext.sourcedir}")
#         logging.info(f"该目录是否存在CMakeLists.txt：{os.path.exists(os.path.join(ext.sourcedir, 'CMakeLists.txt'))}")

#         # 2. 构造CMake参数（适配你的add项目）
#         cmake_args = [
#             # 指定扩展库输出目录（对应CMakeLists.txt的输出路径）
#             f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
#             # 指定Python解释器（避免CMake找错Python）
#             f"-DPython3_EXECUTABLE={sys.executable}",
#             # 可选：自定义参数，你的项目暂时不需要可保留/删除
#             "-DBUILD_PYBIND=True",
#             "-DCMAKE_BUILD_TYPE=Debug",
#             "-DCMAKE_EXPORT_COMPILE_COMMANDS=1"
#         ]
#         # cmake_args = [
#         #     "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir + "/torch_catlass",
#         #     "-DPython3_EXECUTABLE=" + sys.executable,
#         #     "-DBUILD_PYBIND=True"
#         # ]

#         # build_args = []
#         # 3. 构造编译参数（-j是并行编译）
#         build_args = ["-j"]
#         if not os.path.exists(self.build_temp):
#             os.makedirs(self.build_temp)
#         # 5. 执行cmake ..（生成Makefile）
#         # ext.sourcedir是你的项目根目录（包含CMakeLists.txt）
#         logging.info(f"执行cmake命令：cmake {ext.sourcedir} {' '.join(cmake_args)}")
#         subprocess.check_call(
#             ["cmake", ext.sourcedir] + cmake_args,  # 关键：路径改为ext.sourcedir（参考文件是../../，需去掉）
#             cwd=self.build_temp,  # 编译临时目录
#             stdout=subprocess.PIPE,  # 输出重定向到日志
#             stderr=subprocess.STDOUT
#         )

#         # subprocess.check_call(["cmake", os.path.join(ext.sourcedir, "../../")] +
#         #                       cmake_args, cwd=self.build_temp)
#         # 6. 执行cmake --build . --target pybind11_lib（编译目标是CMakeLists.txt中的pybind11_lib）
#         logging.info(f"执行make编译：cmake --build . --target pybind11_lib {' '.join(build_args)}")
#         subprocess.check_call(
#             ["cmake", "--build", ".", "--target", "pybind11_lib"] + build_args,  # 关键：目标改为pybind11_lib（参考文件是_C）
#             cwd=self.build_temp,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT
#         )
#         # subprocess.check_call(
#         #     ["cmake", "--build", ".", "--target", "_C", "-j"] + build_args, cwd=self.build_temp)

#     def generate_pyi(self, ext):
#         # 生成pybind11的类型提示文件（.pyi），方便Python代码补全
#         extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
#         module_name = ext.name  # 你的模块名是add_custom
#         stubgen_args = [module_name, "--output-dir", extdir]
#         # 找到pybind11-stubgen可执行文件
#         stubgen_bin = os.path.join(os.path.dirname(sys.executable), "pybind11-stubgen")
        
#         try:
#             logging.info(f"生成类型提示文件：{stubgen_bin} {' '.join(stubgen_args)}")
#             subprocess.check_call([stubgen_bin] + stubgen_args, cwd=extdir)
#         except FileNotFoundError:
#             logging.warning("未找到pybind11-stubgen，跳过类型提示文件生成（可执行pip install pybind11-stubgen安装）")
#         except subprocess.CalledProcessError as e:
#             logging.warning(f"生成类型提示文件失败：{e}")

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
        IS_NPU = False
    else:
        IS_ROCM = False
        IS_NPU = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
        IS_NPU = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True
        IS_NPU = False
    elif BUILD_TARGET == "npu":
        IS_ROCM = False
        IS_NPU = True

PACKAGE_NAME = "flash_attn"

BASE_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"
)

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
SKIP_CK_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CK_BUILD", "TRUE") == "TRUE" if USE_TRITON_ROCM else False

@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;110;120").split(";")

# 生成 wheel 包的平台标识（适配不同系统）
def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))

# 读取 NVCC 的真实 CUDA 版本
def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

# 生成 NVCC 的-gencode编译参数（核心）
def add_cuda_gencodes(cc_flag, archs, bare_metal_version):
    """
    Adds -gencode flags based on nvcc capabilities:
      - sm_80/90 (regular)
      - sm_100/120 on CUDA >= 12.8
      - Use 100f on CUDA >= 12.9 (Blackwell family-specific)
      - Map requested 110 -> 101 if CUDA < 13.0 (Thor rename)
      - Embed PTX for newest arch for forward compatibility
    """
    # Always-regular 80
    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]

    # Hopper 9.0 needs >= 11.8
    if bare_metal_version >= Version("11.8") and "90" in archs:
        cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]

    # Blackwell 10.x requires >= 12.8
    if bare_metal_version >= Version("12.8"):
        if "100" in archs:
            # CUDA 12.9 introduced "family-specific" for Blackwell (100f)
            if bare_metal_version >= Version("12.9"):
                cc_flag += ["-gencode", "arch=compute_100f,code=sm_100"]
            else:
                cc_flag += ["-gencode", "arch=compute_100,code=sm_100"]

        if "120" in archs:
            # sm_120 is supported in CUDA 12.8/12.9+ toolkits
            if bare_metal_version >= Version("12.9"):
                cc_flag += ["-gencode", "arch=compute_120f,code=sm_120"]
            else:
                cc_flag += ["-gencode", "arch=compute_120,code=sm_120"]


        # Thor rename: 12.9 uses sm_101; 13.0+ uses sm_110
        if "110" in archs:
            if bare_metal_version >= Version("13.0"):
                cc_flag += ["-gencode", "arch=compute_110f,code=sm_110"]
            else:
                # Provide Thor support for CUDA 12.9 via sm_101
                if bare_metal_version >= Version("12.8"):
                    cc_flag += ["-gencode", "arch=compute_101,code=sm_101"]
                # else: no Thor support in older toolkits

    # PTX for newest requested arch (forward-compat)
    numeric = [a for a in archs if a.isdigit()]
    if numeric:
        newest = max(numeric, key=int)
        cc_flag += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]

    return cc_flag

# 获取标准化的 ROCm HIP 版本
def get_hip_version():
    return parse(torch.version.hip.split()[-1].rstrip('-').replace('-', '+'))

# 检查 CUDA 编译工具是否存在（警告而非报错）
def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )

# 检查 ROCm 编译工具是否存在（警告而非报错）
def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found."
    )

# 检测 PyTorch hipify 工具的版本（ROCm 编译关键）
def detect_hipify_v2():
    try:
        from torch.utils.hipify import __version__
        from packaging.version import Version
        if Version(__version__) >= Version("2.0.0"):
            return True
    except Exception as e:
        print("failed to detect pytorch hipify version, defaulting to version 1.0.0 behavior")
        print(e)
    return False

# 给 NVCC 参数添加线程数（加速编译）
def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]

# 将 CPP 文件复制并重命名为 CU（ROCm 编译兼容）
def rename_cpp_to_cu(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")

# ROCm GPU 架构校验函数
def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx950", "gfx942"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"

def find_bisheng_compiler():
    """复刻CMake的find_package(ASC)，定位bisheng编译器和ASC头文件/库"""
    # 从昇腾环境变量定位根目录
    ascend_home = os.getenv("ASCEND_TOOLKIT_HOME", os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend"))
    if not os.path.exists(ascend_home):
        raise RuntimeError(f"昇腾环境未找到！ASCEND_TOOLKIT_HOME={ascend_home}")

    # 定位ASC头文件路径（复刻ASC模块的include）
    asc_include_paths = [
        os.path.join(ascend_home, "compiler/tikcpp/include"),
        os.path.join(ascend_home, "aarch64-linux/tikcpp/include"),
    ]

    # 定位ASC库路径（复刻ASC模块的lib）
    asc_lib_paths = [
        os.path.join(ascend_home, "compiler/lib64"),
        os.path.join(ascend_home, "aarch64-linux/lib64"),
    ]

    return {
        "include_dirs": asc_include_paths,
        "lib_dirs": asc_lib_paths,
        "libs": ["ascendcl"],  # ASC默认链接库
    }

# -------------------------- 3. 自定义BuildExt：直接调用bisheng编译.asc --------------------------
class BishengBuildExt(build_ext):
    def build_extension(self, ext):
        # 核心配置：获取bisheng和依赖路径
        asc_config = find_bisheng_compiler()
        dep_paths = get_dependency_paths()

        # 目标输出路径（复刻add_library的输出）
        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)

        # 构造bisheng编译命令（复刻原CMake所有编译/链接参数）
        compile_cmd = [
            "bisheng",  # 用bisheng编译器（替代g++）
            "-x", "asc",  # 指定语言为ASC（复刻ASC语言编译）
            "--npu-arch=dav-2201",  # 复刻CMAKE的--npu-arch参数
            "-shared",  # 编译为共享库（复刻add_library SHARED）
            "-fPIC",  # 位置无关代码（复刻CMAKE的-fPIC）
            "-std=c++17",  # C++标准（对齐原CMAKE）
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            "-Wl",
            # 头文件路径（复刻target_include_directories）
            *[f"-I{p}" for p in asc_config["include_dirs"]],
            f"-I{dep_paths['python']['include']}",
            f"-I{dep_paths['torch']['include']}",
            f"-I{dep_paths['torch_npu']['include']}",
            # 库路径（复刻target_link_directories）
            *[f"-L{p}" for p in asc_config["lib_dirs"]],
            "-I/root/miniconda3/lib/python3.10/site-packages/pybind11/include",
            "-I/usr/local/Ascend/cann-8.5.0/include",
            "-I/usr/local/Ascend/cann-8.5.0/runtime/include",
            "-I/usr/local/Ascend/cann-8.5.0/include/experiment/runtime",
            "-I/usr/local/Ascend/cann-8.5.0/include/experiment/msprof",
            "-I/root/miniconda3/lib/python3.10/site-packages/torch/include",
            "-I/root/miniconda3/lib/python3.10/site-packages/torch/include/torch/csrc/api/include",
            "-I/home/lxs/flash-attention-main/csrc/catlass/pybind/23_flash_attention_infer",
            "-I/home/lxs/flash-attention-main/csrc/catlass/pybind/common",
            f"-L{dep_paths['torch']['lib']}",
            f"-L{dep_paths['torch_npu']['lib']}",
            "-L/root/miniconda3/lib/python3.10/site-packages/torch/lib",
            "-L/usr/local/Ascend/ascend-toolkit/latest/lib64",
            # 链接库（复刻target_link_libraries）
            *[f"-l{lib}" for lib in asc_config["libs"]],
            "-ltorch_npu",
            "-lplatform",
            "-lregister",
            "-ltiling_api",
            "-lruntime",
            # 源文件 + 输出文件
            *ext.sources,
            "-o", ext_fullpath,
        ]

        # 打印编译命令（方便调试）
        print("=== 执行毕昇编译器命令 ===")
        print(" ".join(compile_cmd))

        # 执行编译（复刻CMake的build过程）
        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print("编译成功！输出：", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"编译失败！错误输出：{e.stderr}")
            raise e


def get_dependency_paths():
    """复刻CMake的Python3/Torch/torch_npu查找逻辑"""
    # 1. Python3路径（复刻find_package(Python3)）
    python_include = sysconfig.get_config_var("INCLUDEPY")
    python_lib = sysconfig.get_config_var("LIBDIR")

    # 2. Torch路径（复刻execute_process + find_package(Torch)）
    torch_cmake_path = torch.utils.cmake_prefix_path
    torch_include = os.path.join(torch_cmake_path, "Torch/include")
    torch_lib = os.path.join(torch_cmake_path, "Torch/lib")

    # 3. torch_npu路径（复刻execute_process + set TORCH_NPU_PATH）
    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib = os.path.join(torch_npu_path, "lib")

    return {
        "python": {"include": python_include, "lib": python_lib},
        "torch": {"include": torch_include, "lib": torch_lib},
        "torch_npu": {"include": torch_npu_include, "lib": torch_npu_lib},
    }

# cmdclass：用于存储自定义的构建命令（后续会绑定bdist_wheel和build_ext的自定义实现）；
# ext_modules：存储要编译的 C++/CUDA 扩展模块列表（CUDAExtension实例），最终传给setup()函数，是编译的核心载体。
cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
# Git 子模块 / 依赖库检查（保证编译依赖完整）
if os.path.isdir(".git"):
    if not SKIP_CK_BUILD:
        subprocess.run(["git", "submodule", "update", "--init", "csrc/composable_kernel"], check=True)
        subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)
elif not IS_NPU:
    if IS_ROCM:
        if not SKIP_CK_BUILD:
            assert (
                os.path.exists("csrc/composable_kernel/example/ck_tile/01_fmha/generate.py")
            ), "csrc/composable_kernel is missing, please use source distribution or git clone"
    else:
        assert (
            os.path.exists("csrc/cutlass/include/cutlass/cutlass.h")
        ), "csrc/cutlass is missing, please use source distribution or git clone"

# CUDA 环境编译分支（非 SKIP_CUDA_BUILD + 非 IS_ROCM）
if not SKIP_CUDA_BUILD and not IS_ROCM and not IS_NPU:
    # 步骤 1：打印 PyTorch 版本，提取主 / 次版本号
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # 检查 CUDA 编译器，校验 CUDA 版本≥11.7
    check_if_cuda_home_none("flash_attn")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.7"):
            raise RuntimeError(
                "FlashAttention is only supported on CUDA 11.7 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )
        # Build -gencode (regular + PTX + family-specific 'f' when available)
        add_cuda_gencodes(cc_flag, set(cuda_archs()), bare_metal_version)
    else:
        # No nvcc present; warnings already emitted above
        pass

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    # 强制设置 C++11 ABI（兼容 PyTorch）
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    # 定义 NVCC 核心编译参数
    nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    # "--ptxas-options=-v",
    # "--ptxas-options=-O2",
    # "-lineinfo",
    # "-DFLASHATTENTION_DISABLE_BACKWARD",
    # "-DFLASHATTENTION_DISABLE_DROPOUT",
    # "-DFLASHATTENTION_DISABLE_ALIBI",
    # "-DFLASHATTENTION_DISABLE_SOFTCAP",
    # "-DFLASHATTENTION_DISABLE_UNEVEN_K",
    # "-DFLASHATTENTION_DISABLE_LOCAL",
    ]

    # 定义 C++17 编译参数，适配 Windows
    compiler_c17_flag=["-O3", "-std=c++17"]
    # Add Windows-specific flags
    if sys.platform == "win32" and os.getenv('DISTUTILS_USE_SDK') == '1':
        nvcc_flags.extend(["-Xcompiler", "/Zc:__cplusplus"])
        compiler_c17_flag=["-O2", "/std:c++17", "/Zc:__cplusplus"]

    # 构建 CUDAExtension 扩展模块
    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=[
                "csrc/flash_attn/flash_api.cpp",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
            ],
            extra_compile_args={
                "cxx": compiler_c17_flag,
                "nvcc": append_nvcc_threads(nvcc_flags + cc_flag),
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn",
                Path(this_dir) / "csrc" / "flash_attn" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )
# ROCm 环境编译分支（非 SKIP_CUDA_BUILD + IS_ROCM）
elif not SKIP_CUDA_BUILD and IS_ROCM:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Skips CK C++ extension compilation if using Triton Backend
    # 代码生成（CK 库的 FMHA 算子）
    if not SKIP_CK_BUILD:
        ck_dir = "csrc/composable_kernel"

        #use codegen get code dispatch
        if not os.path.exists("./build"):
            os.makedirs("build")

        optdim = os.getenv("OPT_DIM", "32,64,128,256")
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "fwd", "--output_dir", "build", "--receipt", "2", "--optdim", optdim], check=True)
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "fwd_appendkv", "--output_dir", "build", "--receipt", "2", "--optdim", optdim], check=True)
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "fwd_splitkv", "--output_dir", "build", "--receipt", "2", "--optdim", optdim], check=True)
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "bwd", "--output_dir", "build", "--receipt", "2", "--optdim", optdim], check=True)

        # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
        # See https://github.com/pytorch/pytorch/pull/70650
        # 适配 PyTorch 生成器头文件路径
        generator_flag = []
        torch_dir = torch.__path__[0]
        if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
            generator_flag = ["-DOLD_GENERATOR_PATH"]

        # 检查 ROCm 编译器，校验 GPU 架构
        check_if_rocm_home_none("flash_attn")
        archs = os.getenv("GPU_ARCHS", "native").split(";")
        validate_and_update_archs(archs)

        if archs != ['native']:
            cc_flag = [f"--offload-arch={arch}" for arch in archs]
        else:
            arch = torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0]
            cc_flag = [f"--offload-arch={arch}"]

        # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
        # torch._C._GLIBCXX_USE_CXX11_ABI
        # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
        if FORCE_CXX11_ABI:
            torch._C._GLIBCXX_USE_CXX11_ABI = True

        # 收集源码文件，适配 hipify v2
        sources = ["csrc/flash_attn_ck/flash_api.cpp",
                "csrc/flash_attn_ck/flash_common.cpp",
                "csrc/flash_attn_ck/mha_bwd.cpp",
                "csrc/flash_attn_ck/mha_fwd_kvcache.cpp",
                "csrc/flash_attn_ck/mha_fwd.cpp",
                "csrc/flash_attn_ck/mha_varlen_bwd.cpp",
                "csrc/flash_attn_ck/mha_varlen_fwd.cpp"] + glob.glob(
            f"build/fmha_*wd*.cpp"
        )

        # Check if torch is using hipify v2. Until CK is updated with HIPIFY_V2 macro,
        # we must replace the incorrect APIs.
        maybe_hipify_v2_flag = []
        if detect_hipify_v2():
            maybe_hipify_v2_flag = ["-DHIPIFY_V2"]

        # 重命名 CPP 文件为 CU（ROCm 兼容）
        rename_cpp_to_cu(sources)

        renamed_sources = ["csrc/flash_attn_ck/flash_api.cu",
                        "csrc/flash_attn_ck/flash_common.cu",
                        "csrc/flash_attn_ck/mha_bwd.cu",
                        "csrc/flash_attn_ck/mha_fwd_kvcache.cu",
                        "csrc/flash_attn_ck/mha_fwd.cu",
                        "csrc/flash_attn_ck/mha_varlen_bwd.cu",
                        "csrc/flash_attn_ck/mha_varlen_fwd.cu"] + glob.glob(f"build/fmha_*wd*.cu")

        # 定义 ROCm 编译参数
        cc_flag += ["-O3","-std=c++20",
                    "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                    "-fgpu-flush-denormals-to-zero",
                    "-DCK_ENABLE_BF16",
                    "-DCK_ENABLE_BF8",
                    "-DCK_ENABLE_FP16",
                    "-DCK_ENABLE_FP32",
                    "-DCK_ENABLE_FP64",
                    "-DCK_ENABLE_FP8",
                    "-DCK_ENABLE_INT8",
                    "-DCK_USE_XDL",
                    "-DUSE_PROF_API=1",
                    # "-DFLASHATTENTION_DISABLE_BACKWARD",
                    "-D__HIP_PLATFORM_HCC__=1"]

        cc_flag += [f"-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={os.environ.get('CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT', 3)}"]

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        # 适配不同 ROCm 版本的编译参数
        hip_version = get_hip_version()
        if hip_version > Version('5.5.00000'):
            cc_flag += ["-mllvm", "--lsr-drop-solution=1"]
        if hip_version > Version('5.7.23302'):
            cc_flag += ["-fno-offload-uniform-block"]
        if hip_version > Version('6.1.40090'):
            cc_flag += ["-mllvm", "-enable-post-misched=0"]
        if hip_version > Version('6.2.41132'):
            cc_flag += ["-mllvm", "-amdgpu-early-inline-all=true",
                        "-mllvm", "-amdgpu-function-calls=false"]
        if hip_version > Version('6.2.41133') and hip_version < Version('6.3.00000'):
            cc_flag += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

        # 构建编译参数字典
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++20"] + generator_flag + maybe_hipify_v2_flag,
            "nvcc": cc_flag + generator_flag + maybe_hipify_v2_flag,
        }

        # 定义头文件目录
        include_dirs = [
            Path(this_dir) / "csrc" / "composable_kernel" / "include",
            Path(this_dir) / "csrc" / "composable_kernel" / "library" / "include",
            Path(this_dir) / "csrc" / "composable_kernel" / "example" / "ck_tile" / "01_fmha",
        ]

        # 构建 CUDAExtension（ROCm 适配）
        ext_modules.append(
            CUDAExtension(
                name="flash_attn_2_cuda",
                sources=renamed_sources,
                extra_compile_args=extra_compile_args,
                include_dirs=include_dirs,
            )
        )
elif not SKIP_CUDA_BUILD and IS_NPU:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    source_files = glob.glob(os.path.join(BASE_DIR, "csrc/catlass/pybind", "*.asc"), recursive=True)

    ext_modules.append(Extension(
        name="flash_attn_2_cuda",  # 模块名（复刻原CMake的custom_ops库名）
        sources=source_files,
        language="c++",  # 基础语言标记（bisheng实际处理ASC）
    ))

# 函数核心：从flash_attn/__init__.py中读取官方版本号，结合本地版本后缀生成最终版本（符合 PEP 440 版本规范）。
def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)

# 生成预编译 wheel 包的 URL 和文件名
def get_wheel_url():
    # 提取基础版本 / 平台信息
    torch_version_raw = parse(torch.__version__)
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    flash_version = get_package_version()
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    if IS_ROCM:
        # ROCm 环境 wheel 文件名生成
        torch_hip_version = get_hip_version()
        hip_version = f"{torch_hip_version.major}{torch_hip_version.minor}"
        wheel_filename = f"{PACKAGE_NAME}-{flash_version}+rocm{hip_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
        # ===================== 新增：NPU分支 =====================
    elif IS_NPU:
        torch_npu_version = parse(torch.__version__)  # torch_npu版本
        # npu_version = get_npu_version()
        npu_ver_tag = "80"
        wheel_filename = f"{PACKAGE_NAME}-{flash_version}+npu{npu_ver_tag}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    else:
        # CUDA 环境 wheel 文件名生成
        # Determine the version numbers that will be used to determine the correct wheel
        # We're using the CUDA version used to build torch, not the one currently installed
        # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
        torch_cuda_version = parse(torch.version.cuda)
        # For CUDA 11, we only compile for CUDA 11.8, and for CUDA 12 we only compile for CUDA 12.3
        # to save CI time. Minor versions should be compatible.
        torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.3")
        # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
        cuda_version = f"{torch_cuda_version.major}"

        # Determine wheel URL based on CUDA version, torch version, python version and OS
        wheel_filename = f"{PACKAGE_NAME}-{flash_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"

    # 拼接下载 URL 并返回
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{flash_version}", wheel_name=wheel_filename)

    return wheel_url, wheel_filename

# 预编译 Wheel 下载命令类
class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        # 强制源码编译判断 若FORCE_BUILD=True（用户手动指定强制源码编译），直接调用父类_bdist_wheel的run()方法，执行默认的源码编译流程，跳过后续下载逻辑。
        if FORCE_BUILD:
            return super().run()

        # 生成 wheel 下载 URL 并打印
        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        try:
            # 尝试下载预编译 wheel（核心 try 块）
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            # 将下载的 wheel 移动到标准输出目录
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        # 下载失败时降级为源码编译
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()

# 智能编译任务数控制类
class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

USE_NINJA = os.getenv('USE_NINJA') == '1'
setup(
    name=PACKAGE_NAME,               # 指定包的正式名称，即之前定义的flash_attn；
    version=get_package_version(),   # 指定包的版本号，调用之前的get_package_version()函数获取
    packages=find_packages(          # 自动递归查找项目中的 Python 包
        exclude=(                    # 排除不需要打包的目录（这些目录仅用于编译 / 测试 / 文档，无需包含在最终安装包中）
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "flash_attn.egg-info",
        )
    ),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",   # description：包的简短描述（一句话），pip show flash_attn时显示；
    long_description=long_description,                                          # long_description：从README.md读取的长描述，PyPI 包页面会展示完整内容；
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",                         # 定义项目的官方仓库地址，元数据字段，方便用户查看源码 / 提交问题
    classifiers=[                                                               # 给包添加分类标签，帮助 PyPI / 用户快速识别包的特性：
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    # 扩展模块配置
    ext_modules=ext_modules,                                                                  # 指定要编译的 C/CUDA 扩展模块列表（即之前构建的CUDAExtension实例）
    cmdclass = {"build_ext": BishengBuildExt},
    # cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": NinjaBuildExtension}           # 根据是否有扩展模块，动态配置自定义构建命令
    # if ext_modules
    # else {
    #     "bdist_wheel": CachedWheelsCommand,
    # },
    python_requires=">=3.9",
    install_requires=[              # 定义包的运行时依赖
        "torch",
        "einops",
    ],
    setup_requires=[                # 定义包的构建时依赖（执行setup.py编译扩展模块时需要的包，运行时无需）
        "packaging",
        "psutil",
        "ninja",
    ],
)
