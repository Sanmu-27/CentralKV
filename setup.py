from setuptools import setup, find_packages

setup(
    name="centralkv",
    version="0.1.0",
    description="Official implementation of CentralKV: Compressing Long-Context LLMs via Attention-Graph Centrality",
    author="Anonymous",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "networkx>=3.1",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "psutil",
        "pynvml",
    ],
)





