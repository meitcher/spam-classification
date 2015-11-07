import os, re, string
import numpy as np


class Dataset:
	
	def __init__(self, dirname, subdirs):
		self.dirname = dirname
		self.subdirs = subdirs
		self.vocabulary = {}
		self.indices_vocabulary = {}
		self.word_indices = {}
		self.indices_word = {}


	def __iter__(self):
		return self.get_text()


	def get_text(self):
		for t, subname in enumerate(self.subdirs):
			for fname in os.listdir(os.path.join(self.dirname, subname)):
				text = open(os.path.join(self.dirname, subname, fname), "r", encoding="utf-8").read()
				yield (self.preprocess(text), t)


	def remove_punctuation(self, s):
		table = s.maketrans("","",string.punctuation)
		return s.translate(table)


	def transform_number(self, text):
		r = re.compile(r"\d")
		return r.sub('NUMBER', text)


	def remove_html(self, text):
		r = re.compile(r"<.+?>")
		return r.sub('', text)


	def transform_urls(self, text):
		r = re.compile(r'(http|https)://[^\s]*')
		return r.sub('URL', text)


	def transform_emails(self, text):
		r = re.compile(r'[^\s]+@[^\s]+')
		return r.sub('EMAIL', text)


	def transform_dollar(self, text):
		r = re.compile(r'[$]+')
		return r.sub('DOLLAR', text)


	def tokenize(self, text):
		return text.strip().split()


	def preprocess(self, text):
		text = text.lower()
		text = self.transform_number(text)
		text = self.transform_urls(text)
		text = self.transform_emails(text)
		text = self.transform_dollar(text)
		text = self.remove_punctuation(text)
		text = self.remove_html(text)
		return self.tokenize(text)


	def build_vocabulary(self, sentence):
		for s in sentence:
			if s not in self.vocabulary:
				self.vocabulary[s] = 0
			self.vocabulary[s] += 1


	def get_word_indices(self):
		if self.word_indices == {}:
			self.word_indices = dict(zip(self.vocabulary.keys(), range(len(self.vocabulary))))
		return self.word_indices


	def get_indices_word(self):
		if self.indices_word == {}:
			self.indices_word = dict(zip(range(len(self.vocabulary)), self.vocabulary.keys()))
		return self.indices_word


	def get_indices_count(self):
		if self.indices_vocabulary == {}:
			self.indices_vocabulary = dict(zip(range(len(self.vocabulary)), self.vocabulary.values()))
		return self.indices_vocabulary


	def bag_of_words(self, id_text): # alert: use a lot of memory!
		Z = np.zeros((len(id_text), len(self.vocabulary)))
		for i, ind_sentence in enumerate(id_text):
			for j in ind_sentence:
				Z[i, j] += 1
		return Z

