import pandas as pd
import os
import warnings
import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)

INPUT_CSV = "regras_condominio.csv"
OUTPUT_CSV = "regras_classificadas.csv"
NUM_CLUSTERS = 8

# Garantir stopwords
try:
    stopwords.words("portuguese")
except:
    nltk.download("stopwords")

# AQUI ESTÁ A CORREÇÃO
stop_pt = stopwords.words("portuguese")   # <<<<<<<<< CORRIGIDO


def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-zA-Z0-9\s]", " ", texto)
    return texto


def run_nlp():
    if not os.path.exists(INPUT_CSV):
        print(f"ERRO: CSV '{INPUT_CSV}' não encontrado.")
        return pd.DataFrame()

    df = pd.read_csv(INPUT_CSV)

    if "Texto_Limpo" not in df.columns:
        print("ERRO: coluna Texto_Limpo ausente.")
        return pd.DataFrame()

    print(f"🔍 Vetorizando {len(df)} regras...")

    textos = df["Texto_Limpo"].fillna("").apply(normalizar)

    vectorizer = TfidfVectorizer(
        max_features=400,
        stop_words=stop_pt,      # Lista válida
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(textos)

    print("🧠 Treinando KMeans...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=20)
    df["Cluster_ID"] = kmeans.fit_predict(X)

    termos = vectorizer.get_feature_names_out()
    centers = kmeans.cluster_centers_.argsort()[:, ::-1]

    cluster_names = {}
    for c in range(NUM_CLUSTERS):
        top_words = [termos[i] for i in centers[c][:5]]
        nome = " | ".join(top_words).title()
        cluster_names[c] = nome
        print(f"Cluster {c}: {top_words} → {nome}")

    df["Categoria_NLP"] = df["Cluster_ID"].map(cluster_names)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\n✅ NLP concluído!")
    print(f"Arquivo gerado: {OUTPUT_CSV}")

    return df


if __name__ == "__main__":
    run_nlp()
