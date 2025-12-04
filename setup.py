from setuptools import setup, find_packages

setup(
    name='tamv-analysis',
    version='0.1.0',
    description='Tense, Aspect, Mood, Voice analysis for corpus linguistics',
    author='TAMV Analysis Project',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'spacy>=3.5.0',
        'nltk>=3.8.0',
        'scikit-learn>=1.2.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'plotly>=5.14.0',
        'convokit>=2.5.0',
        'tqdm>=4.65.0',
    ],
    entry_points={
        'console_scripts': [
            'tamv-analyze=main:main',
        ],
    },
)
