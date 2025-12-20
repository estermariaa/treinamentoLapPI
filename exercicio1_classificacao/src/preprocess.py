import nltk
import unicodedata
import re
import string
import spacy
from nltk.corpus import stopwords

# Certifique-se de que os recursos NLTK estão baixados
nltk.download('stopwords', quiet=True)

# Carregar SpaCy e stopwords uma única vez
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
stopwords_en = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Aplica um pipeline completo de pré-processamento em uma string de texto.
    """

    # Passo 1 - Converter para minúsculas e remover acentuação
    text = text.lower()
    text = ''.join(
        caractere for caractere in unicodedata.normalize('NFD', text)
        if unicodedata.category(caractere) != 'Mn'
    )

    # Passo 2 - Remover menções e hashtags
    text = re.sub(r"[@#][\w@]+", " ", text)

    # Passo 3 - Reduzir repetições de caracteres (ex: "loooove" → "loove")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Passo 4 - Remover URLs
    text = re.sub(r'(http[s]?://\S+|www\.\S+)', '', text, flags=re.IGNORECASE)

    # Passo 5 - Remover pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Passo 6 - Tokenizar e remover stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_en]

    # Rejunta o texto para passar pelo SpaCy
    text = " ".join(tokens)

    # Passo 7 - Remover entidades (organizações, locais)
    doc = nlp(text)
    entidades_para_remover = {"ORG", "GPE", "LOC"}
    tokens_filtrados = [token.text for token in doc if token.ent_type_ not in entidades_para_remover]

    # Atualiza o texto
    text = " ".join(tokens_filtrados)

    # Passo 8 - Remover tokens numéricos
    tokens = text.split()
    tokens = [t for t in tokens if not re.search(r'\d', t)]

    # Passo 9 - Remover palavras curtas (< 3 letras)
    tokens = [t for t in tokens if len(t) >= 3]

    # Passo 10 - Remover nomes de pessoas
    doc = nlp(" ".join(tokens))
    tokens = [token.text for token in doc if token.ent_type_ != "PERSON"]

    # Passo 11 - Lematização final
    doc = nlp(" ".join(tokens))
    tokens_lematizados = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return " ".join(tokens_lematizados)
