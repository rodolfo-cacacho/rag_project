import re
import nltk
from nltk.corpus import stopwords
import spacy
from utils.document_cleaning import post_clean_document


nltk.download('stopwords')
german_stopwords = set(stopwords.words('german'))
domain_stopwords = ['sowie','überblick','beziehungsweise','verschieden','beispiel','gemäß']



def extract_tokens_lemm_stop(text,german_stopwords=german_stopwords,domain_stopwords=domain_stopwords,nlp_model = None):
    
    if nlp_model:
        nlp = spacy.load(nlp_model)
    else:
        nlp = spacy.load('de_core_news_lg')

    # Create spaCy doc
    text = re.sub('\n',' ',text)
    doc = nlp(text)
    char_set = re.escape('.,-')
    # Lemmatization and stop word removal
    tokens = [
        token.lemma_.lower() for token in doc 
        if token.lemma_.lower() not in german_stopwords
        and token.lemma_.lower() not in domain_stopwords
        and not token.is_punct 
        and len(token.lemma_) > 1
        and not token.lemma_.isdigit()
    ]

    # Additional filtering: Remove tokens with only special characters or short numeric-like patterns
    tokens = [
        re.sub(rf'^[{char_set}]|[{char_set}]$', '', token) for token in tokens
        if not re.fullmatch(r'\d{1}|\d+[,\.]?\d*', token)  # Exclude short or formatted numbers
        and not re.fullmatch(r'[^\w]+', token)  # Exclude tokens with only special characters
        and len(re.sub(rf'^[{char_set}]|[{char_set}]$', '', token)) > 1
    ]

    return tokens

def clean_extracted_values_to_tokens(extract_values):
    """
    Function to transform and clean extracted values and words and convert them into tokens

    params:
    - extract_values (dict): Dictionary with different type of extracted values 
    
    """
    tokens_total = []
    for key,vals in extract_values.items():
        if vals:
            if key == 'parentheses_terms':
                for term in vals:
                    clean_text,extracted_terms = post_clean_document(term)
                    clean_text_tokens = extract_tokens_lemm_stop(clean_text)
                    tokens_total.extend(clean_text_tokens)
                    for key_1,val_1 in extracted_terms.items():
                        if val_1:
                            if key_1 == 'parentheses_terms':
                                for term in val_1:
                                    clean_text_1,extracted_terms_1 = post_clean_document(term)
                                    clean_text_1_tokens = extract_tokens_lemm_stop(clean_text_1)
                                    tokens_total.extend(clean_text_1_tokens)
                                    for key_2,val_2 in extracted_terms_1.items():
                                        if val_2:
                                            tokens_total.extend(val_2)
                                            print(f'Wow {key_2}')
                            else:
                                tokens_total.extend(val_1)
            else:
                tokens_total.extend(vals)

    return tokens_total