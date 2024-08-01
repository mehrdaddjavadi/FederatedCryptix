from setuptools import setup, find_packages

setup(
    name='FederatedCryptix',  
    version='1.0.0',
    description='FederatedCryptix is an innovative and modular framework designed for federated learning with a strong focus on cryptographic security.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Mehrdad Javadi',
    author_email='mehrdaddjavadi@gmail.com',
    url='https://github.com/mehrdaddjavadi/FederatedCryptix',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='federated learning, machine learning, deep learning, active learning, encryption, homomorphic encryption, FederatedCryptix',
    install_requires=[
        'numpy',
        'tensorflow',
        'websockets',
        'pytest',
        'tenseal'
    ],
    python_requires='>=3.6',
    include_package_data=True,
    zip_safe=False,
)
