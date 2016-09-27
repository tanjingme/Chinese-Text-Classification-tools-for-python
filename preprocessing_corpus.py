# coding=utf-8
import os,sys
import os.path
import jieba.posseg as pseg
POS_WHITE_LIST=["n"]
def corpus_preprocess(corpus_directory_path,stop_file_path):
	stopfile=open(stop_file_path)
	stopwords=stopfile.read().decode("gbk").split()
	stopfile.close()
	category_names=[items[1] for items in os.walk(corpus_directory_path)][0]
	category_paths=[items[0] for items in os.walk(corpus_directory_path)][1:]
	docs_names=[items[2] for items in os.walk(corpus_directory_path)][1:]
	for i,doc_names in enumerate(docs_names):
		for doc_name in doc_names:
			input=open(category_paths[i]+'\\'+doc_name,"r")
			text=input.read().decode("utf-8")
			words=[w.word for w in pseg.cut(text) if w.word not in stopwords and w.flag in POS_WHITE_LIST]
			if len(words)>1:
				print category_names[i]+" "+doc_name+" "+" ".join(words).encode("gbk")
			input.close()
if __name__=="__main__":
	corpus_preprocess("D:\BaiduYunDownload\experiment\dissertation_experiment\corpus","D:\BaiduYunDownload\experiment\dissertation_experiment\stopwords.txt")