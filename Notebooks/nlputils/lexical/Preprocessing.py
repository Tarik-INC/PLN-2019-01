import nltk
import unidecode
import string

class Preprocessing:
    def __init__(self):
        sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        stemmer = nltk.stem.RSLPStemmer()
    
    def remove_accents(text):
        return unidecode.unidecode(text)
    
    def remove_pontuction(text):
        return text.translate(str.maketrans(" "," ", string.punctuation))
    
    def tokenize_senteces(text):
        sentences = self.sent_tokenizer.tokenize(text)
        return sentences
    
    def tokenize_words(text):
        tokens = nltk.word_tokenize(text)
    
    def lemmatize(text):
        return text
    
    def stemmize(tokens):
        return [self.stemmer.stem(word) for word in tokens]
    
    def normalization_pipeline(text, remove_accents=False, remove_pontuction=False, tokenize_senteces=False, tokenize_words=False, lemmatize=False, stemmize=False ):
        text = self.remove_accents(text) if remove_accents else text
        text = self.remove_pontuction(text) if remove_pontuction else text
        text = self.tokenize_senteces(text) if tokenize_senteces else text
        text = self.tokenize_words(text) if tokenize_words else text
        text = self.lemmatize(text) if lemmatize else text
        text = self.stemmize(text) if stemmize else text
        
