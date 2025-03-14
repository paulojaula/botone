import openpyxl
import nltk
nltk.download('all')  # Mantenha apenas esta linha
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Exibe a imagem betman.png
st.image("betman.png", width=200)

@st.cache_data
def ler_excel(caminho_arquivo):
    """Lê um arquivo Excel e retorna um dicionário com perguntas e respostas."""
    try:
        workbook = openpyxl.load_workbook(caminho_arquivo)
        sheet = workbook.active
        dados = {}
        for row in sheet.iter_rows(min_row=2, values_only=True):
            pergunta, resposta = row
            dados[pergunta] = resposta
        return dados
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho_arquivo}")
        return {}
    except Exception as e:
        st.error(f"Ocorreu um erro ao ler o arquivo Excel: {e}")
        return {}

@st.cache_data
def preprocessar_texto(texto):
    """Pré-processa o texto para PLN."""
    texto = texto.lower()
    tokens = word_tokenize(texto)
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stopwords.words('portuguese')]
    return ' '.join(tokens)

@st.cache_data
def calcular_tfidf_similaridade(perguntas_excel, pergunta_cliente):
    """Calcula a similaridade entre a pergunta do cliente e as perguntas do Excel."""
    perguntas_preprocessadas = [preprocessar_texto(pergunta) for pergunta in perguntas_excel]
    pergunta_cliente_preprocessada = preprocessar_texto(pergunta_cliente)

    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(perguntas_preprocessadas + [pergunta_cliente_