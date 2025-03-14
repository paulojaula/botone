import openpyxl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download dos recursos do NLTK (se necessário)
nltk.download('punkt')
nltk.download('stopwords')

# Exibe a imagem betman.png
st.image("betman.png", width=200)

def ler_excel(caminho_arquivo):
    """Lê um arquivo Excel e retorna um dicionário com perguntas e respostas."""
    workbook = openpyxl.load_workbook(caminho_arquivo)
    sheet = workbook.active
    dados = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):  # Assume que a primeira linha contém os cabeçalhos
        pergunta, resposta = row
        dados[pergunta] = resposta
    return dados

def preprocessar_texto(texto):
    """Pré-processa o texto para PLN."""
    texto = texto.lower()
    tokens = word_tokenize(texto)
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stopwords.words('portuguese')]
    return ' '.join(tokens)

def encontrar_resposta(pergunta_cliente, dados):
    """Encontra a resposta para a pergunta do cliente."""
    perguntas_excel = list(dados.keys())
    perguntas_preprocessadas = [preprocessar_texto(pergunta) for pergunta in perguntas_excel]
    pergunta_cliente_preprocessada = preprocessar_texto(pergunta_cliente)

    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(perguntas_preprocessadas + [pergunta_cliente_preprocessada])
    matriz_similaridade = cosine_similarity(matriz_tfidf[-1], matriz_tfidf[:-1])

    indice_melhor_pergunta = matriz_similaridade.argmax()
    melhor_pergunta = perguntas_excel[indice_melhor_pergunta]

    return dados[melhor_pergunta]

# Interface com Streamlit
st.title("Robô de Apostas")
pergunta_cliente = st.text_input("Faça sua pergunta:")

if pergunta_cliente:
    dados = ler_excel("apostas.xlsx")
    resposta = encontrar_resposta(pergunta_cliente, dados)
    st.write(resposta)