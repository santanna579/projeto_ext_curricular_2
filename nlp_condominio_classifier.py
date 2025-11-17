import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
import warnings

# Ignora warnings do scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURAÇÕES ---
INPUT_CSV = 'regras_condominio.csv'
OUTPUT_CSV = 'regras_classificadas.csv'
# Definimos 6 clusters (categorias) como um bom ponto de partida para regimentos internos.
NUM_CLUSTERS = 6 


def run_condominio_nlp():
    """Executa a vetorização e o clustering (K-Means) nas regras do condomínio."""
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ ERRO: Arquivo de entrada '{INPUT_CSV}' não encontrado. Execute o pdf_parser.py primeiro.")
        return pd.DataFrame()

    print("\n===============================================")
    print("= FASE NLP: CLASSIFICAÇÃO DE REGRAS (K-MEANS) =")
    print("===============================================")

    df = pd.read_csv(INPUT_CSV)
    
    if df['Texto_Limpo'].isnull().all() or df.empty:
        print("❌ ERRO: A coluna 'Texto_Limpo' está vazia. O parser de PDF falhou ao extrair o conteúdo.")
        return pd.DataFrame()

    # 1. Vetorização (TF-IDF)
    # Transforma o texto limpo em vetores numéricos para o algoritmo.
    print(f"Processando {len(df)} regras e vetorizando texto...")
    
    vectorizer = TfidfVectorizer(max_features=150, stop_words='english') # Mantemos um vocabulário limitado
    X = vectorizer.fit_transform(df['Texto_Limpo'])

    # 2. Modelagem (K-Means Clustering)
    print(f"Aplicando K-Means com {NUM_CLUSTERS} clusters...")
    
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', random_state=42, n_init=10)
    # Treina o modelo e atribui um cluster (ID de categoria) a cada regra
    df['Cluster_ID'] = kmeans.fit_predict(X)

    # 3. Análise dos Clusters (Palavras-Chave)
    print("\n--- PASSO CRÍTICO: ANÁLISE PARA NOMEAÇÃO DAS CATEGORIAS ---")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_names = {}
    for i in range(NUM_CLUSTERS):
        # Seleciona as 5 palavras mais importantes de cada cluster
        top_words = [terms[ind] for ind in order_centroids[i, :5]]
        
        # O resultado inicial é a lista de palavras-chave. Jéssica fará a interpretação final.
        cluster_names[i] = f"A Classificar: {', '.join(top_words)}"
        print(f"Cluster {i}: Palavras-chave: {top_words}")
    
    print("----------------------------------------------------------")
    print("🎯 Interpretação: Baseado nas palavras-chave acima, atribua um nome profissional.")
    print("Exemplo: Se o Cluster 0 tiver 'silêncio', 'ruído', 'festa', o nome será 'Regras de Silêncio'.")
    print("----------------------------------------------------------")

    # Mapeamento e Salvamento (Temporário com rótulos brutos)
    df['Categoria_Final'] = df['Cluster_ID'].map(cluster_names)
    
    df_output = df[['Artigo', 'Texto_Regra', 'Tem_Multa', 'Categoria_Final', 'Cluster_ID', 'Texto_Limpo']]
    df_output.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    print(f"\n✅ Sucesso! O arquivo '{OUTPUT_CSV}' foi gerado.")
    print(f"Total de {len(df)} regras classificadas em {NUM_CLUSTERS} categorias.")

    return df_output

if __name__ == "__main__":
    df_classified = run_condominio_nlp()
    
    if not df_classified.empty:
        print("\n--- Amostra das Regras Classificadas ---")
        print(df_classified[['Artigo', 'Categoria_Final', 'Tem_Multa']].head(10))