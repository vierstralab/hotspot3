from setuptools import setup, find_packages

setup(
    name="hotspot3",
    version="1.0.0",
    description="Chromatin accessibility peak caller with an adaptive background model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sergey Abramov, Alexandr Boytsov and Jeff Vierstra",
    url="https://github.com/vierstralab/hotspot3",
    packages=find_packages(where='src'),
    include_package_data=True,
    package_dir={'': 'src'},
    package_data={
        'hotspot3': [
            'scripts/extract_cutcounts.sh',
        ],
    },
    install_requires=[
        "numpy == 1.26.0",
        "scipy",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "hotspot3=hotspot3.cli:main",
            "hotspot3-track-mem=hotspot3.track_memory:main",
            "hotspot3-fdr=hotspot3.multiple_sample_fdr:main",
            "hotspot3-pvals=hotspot3.extract_pvals:main",
        ],
    },
    python_requires=">=3.7",
)
