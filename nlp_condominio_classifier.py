import pandas as pd
import numpy as np
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

INPUT_CSV = "regras_condominio.csv"
OUTPUT_CSV = "regras_classificadas.csv"


# -----------------------------------------------
# 1. NORMALIZAÇÃO DO TEXTO
# -----------------------------------------------
def normalize(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text


# -----------------------------------------------
# 2. CATEGORIAS FIXAS (OFICIAIS)
# -----------------------------------------------
CATEGORIES = {
    "Convivência Geral": ["condominio", "morador", "resident", "conduta", "comportamento"],
    "Silêncio e Horários": ["silencio", "ruido", "som", "festa", "horario"],
    "Uso da Piscina": ["piscina", "traje", "agua", "banho"],
    "Garagem e Estacionamento": ["garagem", "vaga", "estacion", "veiculo", "carro"],
    "Animais de Estimação": ["animal", "pet", "cao", "gato"],
    "Visitantes e Acesso": ["visita", "visitante", "acesso", "entrada", "portaria"],
    "Áreas Comuns": ["area comum", "salão", "quadra", "hall", "corredor"],
    "Segurança e Responsabilidades": ["seguranca", "responsavel", "risco", "chave", "portao"],
    "Limpeza e Manutenção": ["lixo", "limpeza", "manutencao", "sujeira"],
    "Infrações e Penalidades": ["multa", "advertencia", "infracao", "proibido"],
}

CATEGORY_LIST = list(CATEGORIES.keys())


# -----------------------------------------------
# 3. TREINAMENTO AUTOMÁTICO BASEADO EM PALAVRAS-CHAVE
# -----------------------------------------------
def generate_training_data():
    X = []
    y = []

    for category, keywords in CATEGORIES.items():
        for word in keywords:
            X.append(word)
            y.append(category)

    return X, y


# -----------------------------------------------
# 4. CLASSIFICAÇÃO COM MACHINE LEARNING + FALLBACK
# -----------------------------------------------
def classify_text(model, vectorizer, text):
    normalized = normalize(text)
    vector = vectorizer.transform([normalized])

    pred = model.predict(vector)[0]
    prob = max(model.predict_proba(vector)[0])

    # Se o modelo estiver confuso, usa fallback
    if prob < 0.40:
        for category, keywords in CATEGORIES.items():
            if any(k in normalized for k in keywords):
                return category

    return pred


# -----------------------------------------------
# 5. PIPELINE PRINCIPAL
# -----------------------------------------------
def run_nlp():

    df = pd.read_csv(INPUT_CSV)
    df["Texto_Limpo"] = df["Texto_Regra"].astype(str).apply(normalize)

    print("➡️ Treinando modelo NLP...")
    train_x, train_y = generate_training_data()

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_x)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, train_y)

    print("➡️ Classificando regras...")
    df["Assunto_Principal"] = df["Texto_Regra"].apply(
        lambda t: classify_text(model, vectorizer, t)
    )

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ Arquivo gerado: {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    run_nlp()
