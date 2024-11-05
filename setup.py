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
