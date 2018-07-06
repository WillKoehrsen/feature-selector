from setuptools import setup

setup(name="feature_selector",
        version="N/A",
        description="FeatureSelector is a class for removing features for a dataset intended for machine learning",
        url="https://github.com/WillKoehrsen/feature_selector",
        author="Will Koehrsen",
        author_email="wjk@case.edu",
        license="N/A",
        packages=["feature_selector"],
        install_requires=[
            "lightgbm>=2.1.1",
            "matplotlib>=2.1.2",
            "seaborn>=0.8.1",
            "numpy>=1.14.5",
            "pandas>=0.23.1",
            "scikit-learn>=0.19.1"
            ],
        zip_safe=False)
