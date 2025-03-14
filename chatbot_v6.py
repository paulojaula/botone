import openpyxl
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Exibe a imagem super_bet.gif com largura ajustada
col1, col2 = st.columns(2)  # Cria duas colunas
with col1:
    st.image("super_bet.gif", width=500)  # Aumenta o tamanho do GIF para 500 pixels

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
    matriz_tfidf = vectorizer.fit_transform(perguntas_preprocessadas + [pergunta_cliente_preprocessada])
    matriz_similaridade = cosine_similarity(matriz_tfidf[-1], matriz_tfidf[:-1])

    return matriz_similaridade

def encontrar_resposta(pergunta_cliente, dados):
    """Encontra a resposta para a pergunta do cliente."""
    if not dados:
        return "Desculpe, não foi possível carregar os dados."

    perguntas_excel = list(dados.keys())
    matriz_similaridade = calcular_tfidf_similaridade(perguntas_excel, pergunta_cliente)

    indice_melhor_pergunta = matriz_similaridade.argmax()
    melhor_pergunta = perguntas_excel[indice_melhor_pergunta]

    return dados[melhor_pergunta]

with col2:
    st.title("Robô de Apostas")
    pergunta_cliente = st.text_input("Faça sua pergunta:")

    if pergunta_cliente:
        dados = ler_excel("apostas.xlsx")
        resposta = encontrar_resposta(pergunta_cliente, dados)
        st.write(resposta)