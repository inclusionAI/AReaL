# Getting started on arm-based Mac

This README support the installation on the arm based mac. 


## 🚀 Setup on arm based mac.
Our code is implemented in Python. To setup, do the following:
1. Install [Python 3.8.13](https://www.python.org/downloads/release/python-3813/)
2. Install [Java](https://www.java.com/en/download/)
3. Download the source code:
```sh
> git clone https://github.com/princeton-nlp/webshop.git webshop
```
4. Create a virtual environment using [Anaconda](https://anaconda.org/anaconda/python) and activate it
```sh
> conda create -n webshop python=3.8.13
> conda activate webshop
```
5. Install requirements into the `webshop` virtual environment via the `setup.sh` script
```sh
> ./setup_arm.sh [-d small|all]
```




## Default Installation (Failures):
1. `pip3 install -r requirements.txt`

Fails at:
- tokenizers
- nmslib
- lightgbm
- transformers==4.19.2
- PyYAML==6.0.0

Fails, because wrong versions installed
- Werkzeug==2.2.2 (needs to be installed for Flask instead of 3.0.0 to work [https://stackoverflow.com/questions/77213053/why-did-flask-start-failing-with-importerror-cannot-import-name-url-quote-fr])
- numpy-1.24.4 (needs to be installed instead of numpy 1.22 [https://stackoverflow.com/questions/33859531/runtimeerror-module-compiled-against-api-version-a-but-this-version-of-numpy-is])

**tokenizers fix**:
`pip3 install tokenizers`

**nmslib fix**:
`pip3 install Cython`
`pip3 install CFLAGS="-mavx -DWARN(a)=(a)" pip install nmslib`
[https://github.com/nmslib/nmslib/issues/476]

**lightgbm fix**:
[https://github.com/microsoft/LightGBM/issues/5328]
`brew install libomp`
`pip3 install lightgbm`


**transformers fix**:
`pip3 install transformers-4.23.1` works

**PyYAML fix**:
`pip3 install PyYAML==6.0.1` works


## Running setup.sh
Fails at:
- `python -m spacy download en_core_web_lg
`

**Spacy Fix**:
Interestingly:

- `pip install -U 'spacy[apple]'`

Does **NOT** work.

So, first installing spacy with conda works.

Remove:
`spacy==3.3.0` from `requirements.txt`




<!-- env_name:env_conda_webshop -->