import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from streamlit import session_state

# --- CONFIGURAÇÕES DA PÁGINA ---
CONDOMINIO_NOME = "Voraus I"
st.set_page_config(
    page_title=f"Portal de Convivência {CONDOMINIO_NOME}",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caminho para o arquivo final classificado
DATA_URL = 'regras_classificadas.csv'

# ==============================================================================
# 1. FUNÇÕES E ESTILO
# ==============================================================================

# Função para carregar e cachear os dados
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        df.rename(columns={'Categoria_Final': 'Assunto Principal', 'Tem_Multa': 'Penalidade'}, inplace=True)
        # Limpeza e preenchimento
        df['Penalidade'] = df['Penalidade'].astype(str).str.upper().replace({'SIM': 'Multa / Advertência', 'NÃO': 'Sem Penalidade'})
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# Estilo Customizado para Dark Mode e Tabela
st.markdown("""
<style>
/* 1. Títulos e Texto */
h1, h2, h3, h4 {
    color: #f1f5f9; /* Texto Claro */
}
/* 2. Caixa de Destaque (Gatilho Mental) */
.highlight-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1e293b; /* Fundo Secundário Escuro */
    border-left: 6px solid #818cf8; /* Azul Claro de Destaque */
    margin-bottom: 25px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
    color: #f1f5f9;
}
.highlight-box h3 {
    color: #818cf8;
}
/* 3. Tabela - Melhorando o Contraste no Modo Escuro */
.stMarkdown table {
    border-collapse: collapse;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    color: #f1f5f9; /* Texto da Tabela */
}
/* Cabeçalho da Tabela */
.stMarkdown th {
    background-color: #334155 !important; /* Fundo do Cabeçalho */
    color: #ffffff;
    text-align: left;
    padding: 12px 15px;
}
/* Linhas Pares (Zebra Striping) */
.stMarkdown tr:nth-of-type(even) {
    background-color: #1e293b; /* Cor das Linhas Pares */
}
/* Fundo dos Selectboxes na Sidebar */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div {
    background-color: #334155;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. POP-UP DE BOAS-VINDAS (Gatilho Mental de Compromisso)
# ==============================================================================

if 'welcome_popup_shown' not in st.session_state:
    st.session_state.welcome_popup_shown = False

if not st.session_state.welcome_popup_shown:
    with st.empty():
        # Conteúdo do Popup
        st.markdown(f"""
        <div style="background-color: #111827; padding: 30px; border-radius: 10px; border: 2px solid #818cf8; text-align: center; color: white;">
            <h2>🤝 Comprometimento com a Harmonia do {CONDOMINIO_NOME}</h2>
            <p>Este portal foi criado para eliminar dúvidas e garantir a paz em nosso condomínio, usando a Inteligência Artificial para classificar o nosso Regulamento.</p>
            <p>Ao clicar em "Entrar", você se compromete a utilizar as regras com **clareza, respeito e boa-fé**.</p>
            <p><b>Seu uso consciente é o primeiro passo para uma convivência melhor.</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Botão centralizado para fechar o popup
        if st.button("Entrar e Acessar o Mapa de Regras", key="popup_btn"):
            st.session_state.welcome_popup_shown = True
            st.rerun()
    st.stop()
    
# ==============================================================================
# 3. PÁGINA PRINCIPAL (APÓS POPUP)
# ==============================================================================

df = load_data()

st.title(f"🏘️ Portal de Convivência Inteligente {CONDOMINIO_NOME}")

st.markdown(f"""
<div class="highlight-box">
    <h3>🔍 Encontre a Regra Exata em Segundos!</h3>
    <p>Nossa IA classificou todas as cláusulas do Regulamento Interno para que você saiba, imediatamente, como agir e quais regras envolvem penalidades. <b>A clareza elimina conflitos.</b></p>
</div>
""", unsafe_allow_html=True)


# --- SIDEBAR E FILTROS ---
st.sidebar.title("🛠️ Busca Inteligente")

if not df.empty:
    # FILTRO 1: ASSUNTO (Resultado do Clustering)
    categorias = ['Todas'] + sorted(df['Assunto Principal'].unique())
    selected_categoria = st.sidebar.selectbox(
        "🧠 Filtro de Assunto (Classificado pela IA)",
        categorias
    )

    # FILTRO 2: PENALIDADE
    penalidades = ['Todas', 'Multa / Advertência', 'Sem Penalidade']
    selected_penalidade = st.sidebar.selectbox(
        "🚨 Tipo de Penalidade",
        penalidades
    )
    
    # FILTRO 3: BUSCA POR PALAVRA-CHAVE
    search_term = st.sidebar.text_input("🔎 Buscar palavra-chave (Ex: 'pet', 'ruído', 'garagem')")


    # --- APLICA OS FILTROS ---
    df_filtered = df.copy()
    
    if selected_categoria != 'Todas':
        df_filtered = df_filtered[df_filtered['Assunto Principal'] == selected_categoria]
        
    if selected_penalidade != 'Todas':
        df_filtered = df_filtered[df_filtered['Penalidade'] == selected_penalidade]

    if search_term:
        df_filtered = df_filtered[
            df_filtered['Texto_Regra'].astype(str).str.contains(search_term, case=False)
        ]

    
    # --- RESULTADOS E TABELA ---
    st.header(f"Total de Regras Encontradas: {len(df_filtered)}")
    st.markdown("---")

    
    # Função para formatar o link da regra original (se aplicável)
    def format_regra(artigo, texto):
        # Texto da regra em negrito (para destaque)
        return f"<b>{artigo}:</b> {texto}"


    # Prepara o DataFrame para exibição
    df_display = df_filtered[['Artigo', 'Texto_Regra', 'Assunto Principal', 'Penalidade']].copy()
    df_display.columns = ['Artigo', 'Regra Original', 'Assunto Principal (IA)', 'Penalidade']
    
    # Aplica a formatação em HTML para o Artigo e Regra
    df_display['Regra Original'] = df_display.apply(
        lambda row: format_regra(row['Artigo'], row['Regra Original']), axis=1
    )
    
    # Exibe a tabela final
    st.markdown(df_display[['Regra Original', 'Assunto Principal (IA)', 'Penalidade']].to_html(escape=False, index=False), unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption(f"Portal de Convivência {CONDOMINIO_NOME} - Ação Educativa de Ciência de Dados.")

else:
    st.error("Não foi possível carregar os dados. Verifique o arquivo 'regras_classificadas.csv'.")