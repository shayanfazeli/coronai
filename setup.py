from setuptools import setup, find_packages

setup(
    name="coronai",
    version="1.0.0-beta",
    description="CoronAI: A Machine Learning Toolkit for Coronavirus",
    url="https://github.com/shayanfazeli/coronai",
    author="Shayan Fazeli",
    author_email="shayan@cs.ucla.edu",
    license="Apache",
    classifiers=[
          'Intended Audience :: Science/Research',
          #'Development Status :: 1 - Beta',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
    keywords="machine learning,coronavirus,deep learning,inference",
    packages=find_packages(),
    python_requires='>3.6.0',
    scripts=[
        'coronai/bin/coronai_segment2vector',
        'coronai/bin/coronai_unsupervised_clustering',
    ],
    install_requires=[
        'numpy>=1.16.4',
        'numpydoc>=0.9.1',
        'torch>=1.3.0',
        'allennlp'
    ],
    zip_safe=False
)
