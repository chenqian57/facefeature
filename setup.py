import re
import sys
import setuptools
import torch

sys.path.insert(0, "src")

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"

with open("src/metric_trainer/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setuptools.setup(
    name="metric_trainer",
    version=version,
    author="Laughing-q",
    python_requires=">=3.6",
    long_description=long_description,
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    # cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
