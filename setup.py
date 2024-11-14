from setuptools import setup, find_packages

setup(
    name="XequiNet",
    version="0.3.6",
    packages=find_packages(include=["xequinet", "xequinet.*"]),
    include_package_data=True,  # MANIFEST.in
    package_data={
        "xequinet.utils.basis": ["*.dat"],  # basis folder data
        "xequinet.utils.pre_computed": ["*.pt"],  # pre_computed folder data
    },
    install_requires=[
        'torch>=2.0', 
        'torch-geometric>=2.0',
        'torch-cluster',
        'torch-scatter',
        'pytorch-warmup>=0.1',
        'e3nn>=0.5',
        'pydantic>=2.6',
        'ase>=3.22',
        'pyscf>=2.4'
        #==== Extra requirements ====#
        #--------- Geo. Opt. --------#
        # geometric>=1.0
        #----------- PIMD -----------#
        # ipi>=2.6
        #------ Delta Learning ------#
        # tblite>=0.3 # via conda
        # tblite-python>=0.3 # via conda
        # ===========================#
    ],
    python_required='>=3.9',
    entry_points={
        'console_scripts': [
            "xeqtrain = xequinet.run.train:main",
            "xeqjit = xequinet.run.jit_script:main",
            "xeqinfer = xequinet.run.inference:main",
            "xeqtest = xequinet.run.test:main",
            "xeqopt = xequinet.run.geometry:main",
            "xeqmd = xequinet.run.dynamics:main",
            "xeqipi = xequinet.run.pimd:main",
        ]
    },
)
