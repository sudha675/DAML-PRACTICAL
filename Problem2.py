# Import libraries
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet

# Download necessary data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -------------------------------
# STEMMING
# -------------------------------
input_text = "i am still learing python and i am enjoying it"
word_tokens = word_tokenize(input_text)

ps = PorterStemmer()
stemmed_words = [ps.stem(token) for token in word_tokens]

print("Stemmer Words:", stemmed_words)

# -------------------------------
# LEMMATIZATION with POS tagging
# -------------------------------
lemm_text = "I am enjoying learning and running with my friends."
lemm_tokens = word_tokenize(lemm_text)
lemm_pos_tags = pos_tag(lemm_tokens)

# Function to convert POS tags to WordNet format
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for token,pos in lemm_pos_tags:
    wordnet_pos = get_wordnet_pos(pos)
    lemmatized_word = lemmatizer.lemmatize(token, pos=wordnet_pos)
    lemmatized_words.append(lemmatized_word)
print("Lemmatizer Words:", lemmatized_words)
