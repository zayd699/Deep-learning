import pandas
import unicodedata
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl
import re
from pprint import pprint
import streamlit as st


with open('arc-code-ti-publications.pkl', 'rb') as f:
    pubs = pandas.read_pickle(f)

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)
    
def preprocess_text(sen):

    sentence = str(sen)

    # Removing html tags
    sentence = remove_tags(sentence)

    # Remove hyphenation if at the end of a line
    sentence = sentence.replace('-\n', '')

    # Fix ligatures
    sentence = unicodedata.normalize("NFKD", sentence)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

def get_search_result(embedder, text_embeddings, query, closest_n=5):
    query_embedding = embedder.encode([query])
    distances = scipy.spatial.distance.cdist(query_embedding, text_embeddings, "cosine")[0]
    results=[]

    for idx, distance in enumerate(sorted(distances)[:closest_n]):
        results.append(
            {
                "score": 1 - distance,
                "document_id": idx,
                "title": pubs.iloc[idx]["Title"],
                "abstract": pubs.iloc[idx]["Abstract"],
                "abstract_length": pubs.iloc[idx]["Abstract Length"],
                "word_count": pubs.iloc[idx]["Word Count"]
            }
        )
        
    return results

# Process text, create model, and sentences
pubs['Text Processed'] = pubs.apply(lambda row: preprocess_text(row['Text']), axis=1)
pubs['Word Count'] = pubs.apply(lambda row: len(row['Text Processed'].split()), axis=1)
text_df = pubs[['Text Processed',]].copy()
embedder = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
sentences = list(text_df['Text Processed'])

# Eaxmple query sentences
query = ['what is the nearest blackholes of our planet ?']
text_embeddings = embedder.encode(sentences, show_progress_bar=True)


# Streamlit app
st.title('Advanced Search engine for Nasa Publications:')
st.write('A searching Engine developped by Rachid izem - Zayd Bouajaj - Mohamed Wahb chakir - fatima Zahrae Mamouni .')
query = st.text_input('Enter a query to search the publications')
if query:
    results = get_search_result(embedder, text_embeddings, query, closest_n=5)
    st.write('Results:')
    for result in results:
        st.write(result['title'])
        st.write(result['abstract'])
        st.write('Score:', result['score'])
        st.write('Word Count:', result['word_count'])
        st.write('Abstract Length:', result['abstract_length'])
        st.write('---')
