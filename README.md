# Chinese-Text-Classification-tools-for-python
There are some tool functions used for Chinese Text Classification and these functions are implemented with python programme language.
# preprocessing_corpus.py
The python source code file contains the function of preprocessing documents of directories symbolizing categires, and all of the processed documents are shown in the console interface, in which  every line records a document which is expressed by the category of the document (namely the name of directory of the document), document (namely the name of document), the content of the document (namely the remaining words in the document after which is processed with the stopwords are excluded and only nouns are retained)，and each of the items in the line are separated by a space.

USAGE: python preprocessing_corpus.py corpus_path stopword_file > preprocessed_corpus_file
