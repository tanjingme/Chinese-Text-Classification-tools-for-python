# -*- coding: UTF-8 -*-
import math
import sys,os
import random
#sys.path.append("E:\Documents\libsvm-3.21\libsvm-3.21\python")# it add the libsvm packet into your python library
#os.chdir("E:\Documents\libsvm-3.21\libsvm-3.21\python")# do the same
from svmutil import *

class Document:
	def __init__(self,clzz,filename,content):
		self.clzz=clzz
		self.id=clzz+"/"+filename
		self.word_info_pairs={}
		self.word_nums=0
		for w in content:
			self.word_info_pairs.setdefault(w,0)
			self.word_info_pairs[w]+=1
			self.word_nums+=1
	def __str__(self):
		word_info_str=" ".join([items[0]+":"+str(items[1]) for items in sorted(self.word_info_pairs.items(),key=lambda d: d[1],reverse=True)])
		return (self.clzz+" "+self.id+" "+str(self.word_nums)+" "+word_info_str+"\n")
	__repr__=__str__
class Weight:
	corpus=None
	document=None
	@staticmethod
	def compute(word):
		return Weight.tf_idf(word)		
	@staticmethod
	def tf_idf(word):
		return Weight.idf(Weight.corpus,word)*Weight.tf(Weight.document,word)
	@staticmethod
	def set_document(document):
		Weight.document=document
	@staticmethod
	def set_corpus(corpus):
		Weight.corpus=corpus
	@staticmethod
	def idf(corpus,word):
		corpus_size=corpus.corpus_size
		docs_count=corpus.lexicon_docs_counts[word]
		return math.log(1.0*corpus_size/(docs_count+1),10)
	@staticmethod
	def tf(document,word):
		word_nums=document.word_nums
		word_count=document.word_info_pairs[word]
		return word_count*1.0/word_nums
class Corpus:
	def __init__(self,docs_list,categories):
		self.lexicon_docs_counts={}
		self.docs_list=docs_list
		self.corpus_size=len(docs_list)
		self.clzzs_dict=dict(zip(sorted(categories),range(len(categories))))
		for i,doc in enumerate(docs_list):
			for word in doc.word_info_pairs.keys():
				self.lexicon_docs_counts.setdefault(word,0)
				self.lexicon_docs_counts[word]+=1
		
	def compute_weight(self):
		Weight.set_corpus(self)
		for i,doc in enumerate(self.docs_list):
			Weight.set_document(doc)
			for word in doc.word_info_pairs.keys():
				self.docs_list[i].word_info_pairs[word]=Weight.compute(word)
		return self;
	def __str__(self):
		return "The total number of the documents in the corpus is "+str(self.corpus_size)+" and they fall into "+str(len(self.clzzs_dict.keys()));
	__repr__=__str__

class SVMClassifier:
	def __init__(self,train_corpus,test_corpus):
		self.train_corpus=train_corpus
		self.test_corpus=test_corpus
		self.train_y=[]
		self.train_x=[]
		self.test_y=[]
		self.test_x=[]
		self.U=1
		self.L=-1
		self.ranks=[]
		self.feature_words=[]
		self.get_features(10000)
	
	def cbiwdf(self):
		temp={}
		docs_list=self.train_corpus.docs_list
		clzz_nums=len(self.train_corpus.clzzs_dict.keys())
		clzzs_dict=self.train_corpus.clzzs_dict
		for document in docs_list:
			total=sum(document.word_info_pairs.values())
			for word in document.word_info_pairs.keys():
				temp.setdefault(word,[0]*clzz_nums)
				temp[word][clzzs_dict[document.clzz]]+=document.word_info_pairs[word]/total				
		for key in temp.keys():
			temp[key]=max(temp[key])
		self.ranks=sorted(temp.items(),key=lambda d: d[1],reverse=True)
		
	def get_features(self,feature_words_nums):
		self.cbiwdf()
		self.feature_words=[item[0] for item in self.ranks][0:feature_words_nums]
		self.feature_words.sort()
		term_set_dict=dict(zip(self.feature_words,range(len(self.feature_words))))
		clzzs_dict=self.train_corpus.clzzs_dict
		train_docs_list=self.train_corpus.docs_list
		test_docs_list=self.test_corpus.docs_list
		train_min=float('Inf')
		train_max=float('-Inf')
		test_min=float('Inf')
		test_max=float('-Inf')
		for doc in train_docs_list:
			self.train_y.append(clzzs_dict[doc.clzz])
			temp=dict([(term_set_dict[item[0]],item[1])for item in doc.word_info_pairs.items()if term_set_dict.has_key(item[0])])
			train_max=max(temp.values()+[train_max])
			train_min=min(temp.values()+[train_min])
			self.train_x.append(temp)
		for doc in test_docs_list:
			self.test_y.append(clzzs_dict[doc.clzz])
			temp=dict([(term_set_dict[item[0]],item[1])for item in doc.word_info_pairs.items()if term_set_dict.has_key(item[0])])
			test_max=max(temp.values()+[test_max])
			test_min=min(temp.values()+[test_min])
			self.test_x.append(temp)
			
		for d in self.train_x:
			for key in d.keys():
				d[key]=(d[key]-train_min)*(self.U-self.L)/(train_max-train_min)+self.L
		for d in self.test_x:
			for key in d.keys():
				d[key]=(d[key]-test_min)*(self.U-self.L)/(test_max-test_min)+self.L

	def evaluation(self,cmd):
		m=svm_train(self.train_y,self.train_x,cmd)
		p_label,p_acc,p_val=svm_predict(self.test_y,self.test_x,m)
		
		#self.test_status(self.test_y,p_label)# the code is used to show your detail result of the error classification
		return p_acc
	def test_status(self,A,B):
		id_to_clzz={value:key for key, value in self.train_corpus.clzzs_dict.items()}
		temp= filter(lambda x:x[1]!=x[2],zip(range(len(A)),A,B))
		result=[self.test_corpus.docs_list[item[0]].id+" "+id_to_clzz[item[2]]+"/n" for item in temp]
		print result
		
def corpus_allocation(preprocessed_corpus_path,allocation_percentage_list):
	categories=list()
	corpus=[]
	train_corpus=[]
	test_corpus=[]
	documents=[]
	train_documents=[]
	test_documents=[]
	preprocessed_corpus=open(preprocessed_corpus_path,"r")
	for i,it in enumerate(iterator(preprocessed_corpus)):
		clzz=it[0]
		docname=it[1]
		content=it[2]
		doc=Document(clzz,docname,content)
		documents.append(doc)
		if clzz not in categories:
			categories.append(clzz)
			corpus.append([])
		corpus[categories.index(clzz)].append(i)
		
	for i,docs_pos in enumerate(corpus):
		train_corpus.append(docs_pos[0:int(len(docs_pos)*allocation_percentage_list[i])])
		test_corpus.append(docs_pos[int(len(docs_pos)*allocation_percentage_list[i])+1:])

	for i,docs_pos in enumerate(train_corpus):
		train_documents+=[documents[i] for i in docs_pos]
	for i,docs_pos in enumerate(test_corpus):
		test_documents+=[documents[i] for i in docs_pos]
		
	t1=Corpus(train_documents,categories)
	t2=Corpus(test_documents,categories)
	return t1,t2;

	
def iterator(corpus_file):
	l = corpus_file.readline().strip()
	while(l):
		items=l.split()
		yield items[0],items[1],items[2:]
		l = corpus_file.readline().strip()

def random_apportion(b,n):
	v=[random.random() for i in range(n)]
	vsum=sum(v)
	v=[i/vsum for i in v]
	print sum(v)
	print [i/vsum for i in v]
	return v

	

