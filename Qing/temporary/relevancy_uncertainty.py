import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# mhal_llava15_7b_val = f"Qing/result/answer_synthetic_val_data_from_M_HalDetect_modified_llava_15_7b.bin"
mhal_llava15_7b_train = f"result/m_hal/answer_synthetic_train_data_from_M_HalDetect_modified_llava_15_7b.bin"

data = "mhaldetect"
model = "llava15_7b"
split = "train"
def export_text(path):
    with open(path, "rb") as f, open("_".join([data, model, split])+"_tokens.txt", "w") as f_out:
        content = []
        responses = pickle.load(f)

        for idx, resp in responses.items():
            content.append(" ".join(resp['logprobs']['tokens'])+"\n")

        f_out.writelines(content)


# export_text(mhal_llava15_7b_train)


# with open(file_path, 'r') as file:
#         doc_content = file.read()

from sklearn.feature_extraction.text import TfidfVectorizer

token_file = "mhaldetect_llava15_7b_train_tokens.txt"
with open(token_file, 'r') as file:
    doc_content = file.readlines()

def train_tfidf(doc_content):
    # Load the document content
    # token_file = "mhaldetect_llava15_7b_train_tokens.txt"
    # with open(token_file, 'r') as file:
    #     doc_content = file.readlines()

    # Initialize a TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the document content to learn the vocabulary and idf
    vectorizer.fit(doc_content)
    model_filename = 'tfidf_model.joblib'
    joblib.dump(vectorizer, model_filename)
    # The sentence to analyze



def tfidf_encode(vectorizer, sent):
    tfidf_matrix = vectorizer.transform([sent])

    # Convert the TF-IDF matrix for the sentence to a dense format
    dense_tfidf = tfidf_matrix.todense()

    # Get the feature names (vocabulary) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Tokenize the sentence
    tokenized_sentence = sent.split()

    token_weights = []

    # For each token in the input sentence, get its weight from the TF-IDF model
    for token in tokenized_sentence:
        # Check if the token is in the TF-IDF model's vocabulary
        if token in feature_names:
            # Find the index of the token in the feature names
            token_index = list(feature_names).index(token)
            # Append the weight of the token to the list
            token_weights.append(dense_tfidf[0, token_index])
        else:
            # If the token is not found in the model's vocabulary, assign a weight of 0
            token_weights.append(0)

    return token_weights

# def test_tfidf():
#     vectorizer = joblib.load('tfidf_model.joblib')
#     for idx, sent in enumerate(doc_content):
#         # sent = "do you think is going on in this snapshot ? The image dep ict s a large group of people gathered in a grass y field , enjo ying the day by flying k ites . There are several k ites visible in the sky , r anging in size and color , creating a v ibr ant and l ively atmosphere . In addition to the k ites , there are several hand b ags scattered throughout the scene , possibly belonging to some of the people particip ating in the k ite - f lying activity . Some of the hand b ags are closer to the center of the field , while others are position ed more towards the edges . Over all , it appears to be a fun and enjoy able event for everyone involved ."
#         print(idx)
#         # Transform the sentence using the fitted vectorizer
#
#
#         print(len(tokenized_sentence) == len(token_weights))

# train_tfidf(doc_content)
# test_tfidf(doc_content)

def eval_tfidf(doc_content):
    sent = 'The image depicts a large group of people gathered in a grassy field, enjoying the day by flying kites.'

    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the document content to learn the vocabulary and idf
    vectorizer.fit(doc_content)

    new_vectorizer = joblib.load('tfidf_model.joblib')

    # print(tfidf_encode(vectorizer, sent)[:30])
    # print(tfidf_encode(new_vectorizer, sent)[:30])
    sent_weights = tfidf_encode(new_vectorizer, sent)
    print(len(sent.split()))
    print(len(sent_weights))

    for ele in zip(sent.split(), sent_weights):
        print(ele)


eval_tfidf(doc_content)
