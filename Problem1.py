#Upper case 
input = "centurion university technology and management . I am a student of this university ."
input = input.upper()
print(input)

# tokenzation
import nltk
nltk.download('punkt')
from nltk import word_tokenize, sent_tokenize
word = word_tokenize(input)
print(word)
word1=sent_tokenize(input)
print(word1)

#stop words
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')    
# print(stopwords.fileids())
word=input.lower()
word=word_tokenize(word)
stop_words=set(stopwords.words('english'))
filter_word=[]
for token in word:
    if token  not in stop_words:
        filter_word.append(token)
print(filter_word)