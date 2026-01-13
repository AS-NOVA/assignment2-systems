# CS336 Spring 2025 Assignment 2: Systems

## Setup

You can verify that the code from the `cs336-basics` package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## nsys on wsl 使用方案3

以下信息参考自`https://forums.developer.nvidia.com/t/nsys-doesnt-show-cuda-kernel-and-memory-data/315536/8`

nsys工具在进行分析时，例如运行`uv run nsys profile -o result python ./cs336_systems/benchmarking.py `，生成的报告文件里面会缺失gpu信息，这是因为wsl的虚拟化导致cpu和gpu的时间无法对齐。一个办法是让cupti来进行这个时间戳的转换，这比nsys的实现精度低，但至少在wsl上能够正确工作。

首先，保证nsys的版本高于2024.7。

> 通过`apt list -a "nsight-systems*"`，发现我的本机wsl中存在两个nsys，一个新一个旧，其中较新的名为`nsight-systems-2025.3.2`，足够新
> 通过`dpkg -L nsight-systems-2025.3.2 | grep bin/nsys`，找到新的nsys的位置
> 此时`/opt/nvidia/nsight-systems/2025.3.2/bin/nsys --version`确实输出较新的版本号
> 为了尽量减少影响，可以在终端里需要运行nsys时指定`alias nsys='/opt/nvidia/nsight-systems/2025.3.2/bin/nsys'`（重启终端会失效）
> 这样保证每次使用都是用新的那个，并且不扰乱环境

确定了nsys的版本足够新之后，通过`nsys -z`找到nsys的配置文件路径（可能还不存在，此时需手动创建该路径和文件），在该文件中写入
```
CuptiUseRawGpuTimestamps=false
```
这样一行内容。然后重新运行nsys profile，观察结果。

