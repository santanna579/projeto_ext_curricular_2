import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import re

# Baixar recursos do NLTK (só é necessário na primeira execução)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 1. CARREGAMENTO DOS DADOS ---
try:
    # Carrega o CSV que você criou
    df = pd.read_csv('regras_condominio.csv')
    print("Dados carregados com sucesso!")
except FileNotFoundError:
    print("Erro: O arquivo 'regras_condominio.csv' não foi encontrado. Por favor, crie-o conforme o modelo.")
    # Cria um DataFrame vazio para não quebrar o código
    df = pd.DataFrame(columns=['Artigo', 'Texto_Regra', 'Assunto_Bruto', 'Tem_Multa'])
    
if df.empty:
    exit()

# --- 2. PRÉ-PROCESSAMENTO NLP (LIMPEZA DE TEXTO) ---

# Define as stopwords em português (palavras irrelevantes: de, a, o, para)
stop_words_pt = set(stopwords.words('portuguese'))

def clean_text(text):
    """Função para limpar e normalizar o texto para o NLP."""
    # 1. Converte para minúsculas
    text = text.lower()
    
    # 2. Remove pontuações (exceto o que separa palavras)
    text = re.sub(f'[{string.punctuation}]', '', text)
    
    # 3. Remove números
    text = re.sub(r'\d+', '', text)
    
    # 4. Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words_pt]
    
    # Junta o texto novamente
    text = ' '.join(words)
    
    return text

# Aplica a função de limpeza na coluna de regras
df['Texto_Limpo'] = df['Texto_Regra'].apply(clean_text)

# Exibe o resultado da limpeza para verificação
print("\nPré-processamento Concluído. Amostra do Texto Limpo:")
print(df[['Texto_Regra', 'Texto_Limpo']].head())