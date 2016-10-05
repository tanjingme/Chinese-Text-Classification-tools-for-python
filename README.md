# Chinese-Text-Classification-tools-for-python
There are some tool functions used for Chinese Text Classification and these functions are implemented with python programme language.
# preprocessing_corpus.py
The python source code file contains the function of preprocessing documents of directories symbolizing categires, and all of the processed documents are shown in the console interface, in which  every line records a document which is expressed by the category of the document (namely the name of directory of the document), document (namely the name of document), the content of the document (namely the remaining words in the document after which is segmented with the jieba word segmentation fuction and then is processed with the stopwords are excluded and only nouns are retained)，and each of the items in the line are separated by a space. You can also use the output redirection command operator '>' to put the data into a file.

USAGE: 

       python preprocessing_corpus.py corpus_directory_path stopword_file_path > preprocessed_corpus_file

# text_classify.py
The function corpus_allocation() allocate the corpus data into training corpus and testing corpus by setting the proportional list which specifies the share of  documents in specific category. Its input parameters include the string of path of the above preprocessed file and  the proportional list, in which every element must belong to the interval [0,1] and the sum of the elements can be bigger than one. Its return list includes training corpus and testing corpus which are objects of Class Corpus.
corpus_allocation

USAGE: 
       
       train_corpus,test_corpus=corpus_allocation(preprocessed_corpus_path,allocation_percentage_list):

The function compute_weight computing the weight of words corresponding to documents in a corpus and it is the member fuction of the class Corpus. It is a non-parameter function and doesn't return anything. you can change the code in the class Weight code block to change the way how it  implements and now it uses the tf-idf , is also having the code block of how to reduce data dimention.

USAGE:        

       corpus.compute_weight()

The class SVMClassifier use the libsvm function packet to classify the texts. and its input parameter list includes two Corpus object:the first one is training corpus object and the second one is testing corpus object. and its member function evaluation() using to evaluate the model recieves the same control parameter string with libsvm and return the model of accurate.

USAGE: 

       svmcl=SVMClassifier(train_corpus,test_corpus)  
       svmcl.evaluation('-h 0 -c 56.0038 -g 0.0200 -q')

