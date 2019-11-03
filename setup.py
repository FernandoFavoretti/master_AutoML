import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='fangorn',  
     version='0.1',
     packages=setuptools.find_packages(),
     author="fernando.favoretti",
     author_email="fernando.prado@usp.br",
     description="automl package ",
     long_description=long_description,
     install_requires=[
          'pandas',
          'numpy',
          'lightgbm',
          'xgboost',
          'scikit-learn',
     ],
    package_data={'': ['*.ini']},
    include_package_data=True,
   long_description_content_type="text/markdown",
     url="https://github.com/FernandoFavoretti/master_AutoML",
     classifiers=[
         "Programming Language :: Python :: 3",
     ],
 )