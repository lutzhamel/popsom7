from setuptools import setup, find_packages

setup(
    name='popsom7',
    version='7.1.2',
    author='Lutz Hamel',
    author_email='lutz.hamel@gmail.com',
    description="A Fast, User-friendly Implementation of Self-Organizing Maps (SOMs)",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lutzhamel/popsom7',  # update with your repository URL
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'matplotlib',
        'scipy'
    ],
)
