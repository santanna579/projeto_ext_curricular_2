import pandas as pd
import re
import os
import pypdf
import nltk
from nltk.corpus import stopwords
import string
import warnings

# Ignorar warnings sobre recursos experimentais do pypdf, se houver
warnings.filterwarnings("ignore")

# Nome do arquivo de entrada e saída
PDF_FILENAME = '024 - REGULAMENTO INTERNO - reformulado.pdf'
OUTPUT_CSV_FILENAME = 'regras_condominio.csv'

# --- 1. CONFIGURAÇÃO DE PALAVRAS-CHAVE ---
# Palavras-chave em português para detectar se a regra possui penalidade
PENALTY_KEYWORDS = [
    "multa", 
    "advertência", 
    "penalidade", 
    "sanção", 
    "suspensão",
    "sujeita a",
    "incide em",
    "será punido"
]

# --- 2. FUNÇÕES DE EXTRAÇÃO ---

def extract_text_from_pdf(pdf_path):
    """Extrai texto bruto de todas as páginas do PDF."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo '{pdf_path}' não encontrado. Salve o PDF no mesmo diretório do script.")
        return None
    except Exception as e:
        print(f"❌ Ocorreu um erro ao ler o PDF: {e}")
        return None

def parse_regulations(raw_text):
    """
    Usa Regex para identificar e separar Artigos e Cláusulas, e classifica penalidades.
    Este regex é customizado para o formato "Art. Xº" ou "X.Y"
    """
    
    # Padroniza quebras de linha e múltiplos espaços para facilitar o Regex
    text = raw_text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # Padrão de identificação de regras:
    # Captura Art. Xº (ou Art. X), ou X.Y (para subitens)
    # A flag re.I ignora case (Art. e art.)
    # O padrão busca por um marcador seguido por um espaço/hífen e o texto da regra
    # Adiciona-se uma quebra de linha artificial para garantir que o regex separe corretamente
    rule_markers = re.compile(r'(\s*Art\. \d+º\s*-?\s*|\s*Art\. \d+\s*-?\s*|\s*\d+\.\d+\s*-?\s*)', re.I)
    
    # Usamos o marcador para quebrar o texto
    parts = rule_markers.split(text)
    
    regulations = []
    
    # Recombina as partes, buscando por marcadores válidos
    for i in range(1, len(parts), 2):
        marker = parts[i].strip()
        content = parts[i+1].strip()
        
        # Ignora marcadores que não parecem ser o Artigo/Item (como "Página X de Y" ou "CAPÍTULO")
        if not re.match(r'^(Art\.\s*\d+º?|\d+\.\d+)$', marker, re.I):
            continue
            
        # Remove o hífen se ele estiver no marcador
        article_id = marker.replace('-', '').strip()

        # Classificação de Multa
        tem_multa = 'NÃO'
        content_lower = content.lower()
        if any(keyword in content_lower for keyword in PENALTY_KEYWORDS):
            tem_multa = 'SIM'

        regulations.append({
            'Artigo': article_id,
            'Texto_Regra': content,
            'Assunto_Bruto': 'A Classificar (NLP)',
            'Tem_Multa': tem_multa
        })

    return regulations

# --- 3. EXECUÇÃO PRINCIPAL E GERAÇÃO DO CSV ---

if __name__ == "__main__":
    print(f"⏳ Iniciando a extração do Regimento: {PDF_FILENAME}")
    
    raw_text = extract_text_from_pdf(PDF_FILENAME)
    
    if raw_text:
        regulations_list = parse_regulations(raw_text)
        
        if not regulations_list:
            print("⚠️ AVISO: Nenhuma regra foi extraída. O formato do PDF pode ter impedido o Regex.")
        
        df_regulations = pd.DataFrame(regulations_list)
        
        # --- LIMPEZA DE TEXTO PARA O NLP (REAPROVEITAMENTO DO CÓDIGO ANTERIOR) ---
        stop_words_pt = set(stopwords.words('portuguese'))
        
        def clean_text_for_nlp(text):
            text = text.lower()
            text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
            text = re.sub(r'\d+', '', text)
            words = text.split()
            words = [word for word in words if word not in stop_words_pt]
            return ' '.join(words)

        if not df_regulations.empty:
            df_regulations['Texto_Limpo'] = df_regulations['Texto_Regra'].apply(clean_text_for_nlp)

            # Reordena e salva o CSV
            df_regulations = df_regulations[['Artigo', 'Texto_Regra', 'Texto_Limpo', 'Assunto_Bruto', 'Tem_Multa']]
            df_regulations.to_csv(OUTPUT_CSV_FILENAME, index=False, encoding='utf-8')
            
            print("✅ Sucesso! O arquivo regras_condominio.csv foi gerado.")
            print(f"Total de {len(df_regulations)} regras extraídas. Recomenda-se uma revisão humana.")
            print("\nPrimeiras 5 regras extraídas:")
            print(df_regulations.head())