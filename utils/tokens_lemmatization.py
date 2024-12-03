import re
import nltk
from nltk.corpus import stopwords
from utils.document_cleaning import post_clean_document


nltk.download('stopwords')
german_stopwords = set(stopwords.words('german'))
domain_stopwords = ['sowie','überblick','beziehungsweise','verschieden','beispiel','gemäß']

def extract_tokens_lemm_stop(text, nlp,german_stopwords=german_stopwords, domain_stopwords=domain_stopwords):
    """
    Extract lemmatized tokens from text while removing stopwords, punctuation, and short/invalid tokens.
    
    Parameters:
        text (str): Input text to process.
        german_stopwords (set): Set of German stopwords.
        domain_stopwords (set): Set of domain-specific stopwords.
        nlp (spacy.Language): Preloaded spaCy NLP model.

    Returns:
        list: Processed list of tokens.
    """
    # Precompile regex patterns for efficiency
    char_set = re.escape('.,-')
    trim_special_chars = re.compile(rf'^[{char_set}]|[{char_set}]$')
    exclude_numbers = re.compile(r'\d{1}|\d+[,\.]?\d*')
    exclude_special_chars = re.compile(r'[^\w]+')

    # Replace newline characters and create spaCy doc
    text = text.replace('\n', ' ')
    doc = nlp(text)

    # Generate tokens with filtering and lemmatization
    tokens = [
        trim_special_chars.sub('', token.lemma_.lower())
        for token in doc
        if token.lemma_.lower() not in german_stopwords
        and token.lemma_.lower() not in domain_stopwords
        and not token.is_punct
        and not token.lemma_.isdigit()
        and len(token.lemma_) > 1
    ]

    # Additional filtering for invalid tokens
    tokens = [
        token for token in tokens
        if len(token) > 1
        and not exclude_numbers.fullmatch(token)
        and not exclude_special_chars.fullmatch(token)
    ]

    return tokens

def clean_extracted_values_to_tokens(extract_values,nlp):
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
                    clean_text_tokens = extract_tokens_lemm_stop(clean_text,nlp=nlp)
                    tokens_total.extend(clean_text_tokens)
                    for key_1,val_1 in extracted_terms.items():
                        if val_1:
                            if key_1 == 'parentheses_terms':
                                for term in val_1:
                                    clean_text_1,extracted_terms_1 = post_clean_document(term)
                                    clean_text_1_tokens = extract_tokens_lemm_stop(clean_text_1,nlp=nlp)
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