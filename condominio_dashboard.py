# streamlit_portal_full.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import re
import unicodedata
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# -----------------------
# CONFIG
# -----------------------
CONDOMINIO_NOME = "Voraus I"
DATA_CSV = "regras_classificadas.csv"  # output do seu pipeline NLP
st.set_page_config(
    page_title=f"Portal de Convivência {CONDOMINIO_NOME}",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------
# UTILIDADES
# -----------------------
@st.cache_data
def load_data(path=DATA_CSV):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize possible column name variants
    rename_map = {}
    if 'Categoria_NLP' in df.columns and 'Assunto Principal' not in df.columns:
        rename_map['Categoria_NLP'] = 'Assunto Principal'
    if 'Categoria_Final' in df.columns and 'Assunto Principal' not in df.columns:
        rename_map['Categoria_Final'] = 'Assunto Principal'
    if 'Assunto_Principal' in df.columns and 'Assunto Principal' not in df.columns:
        rename_map['Assunto_Principal'] = 'Assunto Principal'
    if 'Texto_Limpo' not in df.columns and 'Texto_Limp' in df.columns:
        rename_map['Texto_Limp'] = 'Texto_Limpo'
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Ensure required columns exist
    for c in ['Artigo', 'Item', 'Texto_Regra', 'Texto_Limpo', 'Assunto Principal', 'Tem_Multa']:
        if c not in df.columns:
            df[c] = ""
    # Friendly penalty label
    df['Penalidade'] = df['Tem_Multa'].astype(str).str.upper().replace({'SIM': 'Multa / Advertência', 'NÃO': 'Sem Penalidade'})
    df['Assunto Principal'] = df['Assunto Principal'].fillna("Sem Categoria")
    return df

def download_link_df(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False, encoding='utf-8')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download CSV</a>'
    return href

def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def highlight_terms(text, query):
    if not query or not isinstance(text, str):
        return text
    # escape regex and highlight tokens (case-insensitive)
    tokens = [t for t in re.split(r'\s+', query.strip()) if t]
    out = text
    for t in tokens:
        if not t:
            continue
        # build case-insensitive pattern, but do not normalize accents here to keep original display
        pattern = re.compile(re.escape(t), flags=re.IGNORECASE)
        out = pattern.sub(lambda m: f"<mark style='background:#fff176;color:#111'>{m.group(0)}</mark>", out)
    return out

def semantic_search_corpus(tfidf_vectorizer, matrix, query, top_k=6):
    if not isinstance(query, str) or query.strip() == "":
        return np.array([], dtype=int), np.array([])
    q = normalize_text(query)
    q_vec = tfidf_vectorizer.transform([q])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return top_idx, sims[top_idx]

# -----------------------
# LOAD DATA
# -----------------------
df = load_data()

# -----------------------
# STYLE (dark-friendly, responsive)
# -----------------------
st.markdown(
    """
    <style>
    /* page background */
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #07101a 100%);
        color: #e6eef8;
    }
    .sidebar .sidebar-content {
        background: #0b1220;
    }
    /* popup-like header card */
    .welcome-card {
        background: linear-gradient(90deg,#0f172a,#07101a);
        border: 1px solid rgba(129,140,248,0.12);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 18px;
    }
    .welcome-card h1 { color: #fff; margin:0; }
    .welcome-card p { color: #cbd5e1; margin:4px 0 0 0; }
    mark { background: #fff176; color:#111 !important; padding:0 2px; border-radius:2px; }
    .result-card { background:#0f172a; color:#e6eef8; padding:12px; border-radius:10px; border:1px solid #1f2937; margin-bottom:12px; }
    .small-muted { color:#94a3b8; font-size:13px; }
    @media (max-width: 800px) {
      .welcome-card { padding: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# SIDEBAR: NAV
# -----------------------
st.sidebar.title("🔧 Controles")
page = st.sidebar.radio("Navegar", ["Dashboard", "Mapa de Regras", "Estatísticas", "Download Center", "Busca Semântica"])
st.sidebar.markdown("---")
st.sidebar.caption("Versão: 1.0 • Portal Profissional")

# -----------------------
# WELCOME AREA (top)
# -----------------------
if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    cols = st.columns([1, 2])
    with cols[0]:
        st.image("https://images.pexels.com/photos/4247766/pexels-photo-4247766.jpeg", width=160)
    with cols[1]:
        st.markdown(f"<div class='welcome-card'><h1>🤝 Portal de Convivência — {CONDOMINIO_NOME}</h1><p>Informação correta e acessível para decisões rápidas e justas. Use o portal para localizar regras, verificar penalidades e baixar relatórios.</p></div>", unsafe_allow_html=True)
    if st.button("Entrar e Explorar o Portal"):
        st.session_state.entered = True
        st.experimental_rerun()
    st.stop()

# -----------------------
# DASHBOARD
# -----------------------
if page == "Dashboard":
    st.title(f"📊 Dashboard - {CONDOMINIO_NOME}")
    if df.empty:
        st.warning("Arquivo CSV não encontrado. Gere 'regras_classificadas.csv' com o pipeline e reabra a página.")
        st.stop()

    total_rules = len(df)
    total_articles = df['Artigo'].nunique()
    total_multas = (df['Tem_Multa'].astype(str).str.upper() == 'SIM').sum()
    total_categories = df['Assunto Principal'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Regras", total_rules)
    c2.metric("Artigos (únicos)", total_articles)
    c3.metric("Regras com Penalidade", total_multas)
    c4.metric("Categorias (IA)", total_categories)

    st.markdown("---")
    st.subheader("📦 Distribuição por Categoria (Top 15)")
    cat_counts = df['Assunto Principal'].value_counts().nlargest(15).reset_index()
    cat_counts.columns = ['Categoria', 'Quantidade']
    if not cat_counts.empty:
        fig_bar = px.bar(cat_counts, x='Quantidade', y='Categoria', orientation='h', color='Quantidade', color_continuous_scale='Blues', template='plotly_dark')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Sem categorias para exibir.")

    st.markdown("---")
    st.subheader("🚨 Penalidades")
    penal = df.groupby(['Assunto Principal', 'Penalidade']).size().reset_index(name='n')
    if not penal.empty:
        fig_pen = px.bar(penal, x='n', y='Assunto Principal', color='Penalidade', barmode='stack', orientation='h', template='plotly_dark', color_discrete_map={"Multa / Advertência": "#ef4444", "Sem Penalidade": "#3b82f6"})
        st.plotly_chart(fig_pen, use_container_width=True)
    else:
        st.info("Sem dados de penalidade.")

    st.markdown("---")
    st.subheader("🧾 Termos mais relevantes")
    all_text = " ".join(df['Texto_Limpo'].fillna("").astype(str).tolist())
    if all_text.strip() == "":
        st.info("Texto limpo ausente — execute o parser primeiro.")
    else:
        vector = TfidfVectorizer(max_features=40)
        try:
            v = vector.fit_transform([all_text])
            terms = vector.get_feature_names_out()
            st.write(", ".join(terms[:20]))
        except Exception:
            st.info("Não foi possível extrair termos (dados insuficientes).")

# -----------------------
# MAPA DE REGRAS
# -----------------------
elif page == "Mapa de Regras":
    st.title("📚 Mapa de Regras")
    if df.empty:
        st.warning("CSV ausente.")
        st.stop()

    col1, col2 = st.columns([3,1])
    with col1:
        cat_filter = st.selectbox("Filtrar por Categoria (IA)", options=['Todas'] + sorted(df['Assunto Principal'].unique()))
    with col2:
        penal_filter = st.selectbox("Filtrar por Penalidade", options=['Todas','Multa / Advertência','Sem Penalidade'])

    search_q = st.text_input("🔎 Buscar na regra (texto livre) — termos destacados", key="map_search")

    df_map = df.copy()
    if cat_filter != 'Todas':
        df_map = df_map[df_map['Assunto Principal'] == cat_filter]
    if penal_filter != 'Todas':
        df_map = df_map[df_map['Penalidade'] == penal_filter]
    st.markdown(f"**Total:** {len(df_map)} regras encontradas")

    cats = df_map['Assunto Principal'].fillna("Sem Categoria").unique()
    for c in sorted(cats):
        sub = df_map[df_map['Assunto Principal'] == c]
        with st.expander(f"📂 {c} — {len(sub)} regras", expanded=False):
            for idx, row in sub.iterrows():
                art = f"{row.get('Artigo','')}{(' ' + str(row.get('Item','')) ) if str(row.get('Item','') ) not in ['','nan','None'] else ''}"
                texto = row.get('Texto_Regra','')
                highlighted = highlight_terms(texto, search_q) if search_q else texto
                st.markdown(f"**{art}** — {row.get('Penalidade','')}", unsafe_allow_html=True)
                st.markdown(f"<div style='padding:10px;border-radius:8px;background:linear-gradient(90deg,#071021,#081227);border:1px solid #111827'>{highlighted}</div>", unsafe_allow_html=True)
                st.markdown("---")

# -----------------------
# ESTATÍSTICAS
# -----------------------
elif page == "Estatísticas":
    st.title("📈 Estatísticas do Regulamento")
    if df.empty:
        st.warning("CSV ausente.")
        st.stop()

    cat_counts = df['Assunto Principal'].value_counts().reset_index()
    cat_counts.columns = ['Categoria','Quantidade']
    if not cat_counts.empty:
        fig = px.pie(cat_counts, names='Categoria', values='Quantidade', title="Distribuição por Categoria (IA)", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    penal = df.groupby(['Assunto Principal','Penalidade']).size().reset_index(name='n')
    if not penal.empty:
        fig2 = px.bar(penal, x='n', y='Assunto Principal', color='Penalidade', orientation='h', barmode='stack', template='plotly_dark', color_discrete_map={"Multa / Advertência": "#ef4444", "Sem Penalidade": "#3b82f6"})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Termos mais frequentes por Categoria (Top 6)")
    try:
        vectorizer = TfidfVectorizer(max_features=400, ngram_range=(1,2))
        docs = df.groupby('Assunto Principal')['Texto_Limpo'].apply(lambda x: " ".join(x.fillna("").astype(str))).reset_index()
        X = vectorizer.fit_transform(docs['Texto_Limpo'])
        terms = vectorizer.get_feature_names_out()
        for i, row in docs.iterrows():
            cat = row['Assunto Principal']
            vec = X[i].toarray().flatten()
            top_idx = np.argsort(-vec)[:6]
            top_words = [terms[j] for j in top_idx]
            st.write(f"**{cat}**: {', '.join(top_words)}")
    except Exception:
        st.info("Não foi possível calcular termos por categoria (dados insuficientes).")

# -----------------------
# DOWNLOAD CENTER
# -----------------------
elif page == "Download Center":
    st.title("📥 Download Center")
    if df.empty:
        st.warning("CSV ausente.")
        st.stop()

    st.markdown("Baixe o CSV completo ou o subset filtrado.")
    st.markdown(download_link_df(df, "regras_classificadas_full.csv"), unsafe_allow_html=True)

    st.markdown("---")
    st.write("Gerar exportação (CSV) com filtros aplicados:")
    sel_cat = st.multiselect("Categorias", options=sorted(df['Assunto Principal'].unique()))
    sel_pen = st.multiselect("Penalidade", options=sorted(df['Penalidade'].unique()))
    q = st.text_input("Filtro texto (opcional)")

    df_dl = df.copy()
    if sel_cat:
        df_dl = df_dl[df_dl['Assunto Principal'].isin(sel_cat)]
    if sel_pen:
        df_dl = df_dl[df_dl['Penalidade'].isin(sel_pen)]
    if q:
        df_dl = df_dl[df_dl['Texto_Regra'].str.contains(q, case=False, na=False)]

    st.markdown(f"Regras selecionadas: {len(df_dl)}")
    st.markdown(download_link_df(df_dl, "regras_exportadas.csv"), unsafe_allow_html=True)

# -----------------------
# BUSCA SEMÂNTICA
# -----------------------
elif page == "Busca Semântica":
    st.title("🔎 Busca Semântica (local, rápido)")
    if df.empty:
        st.warning("CSV ausente.")
        st.stop()

    corpus = df['Texto_Limpo'].fillna("").astype(str).apply(normalize_text)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    try:
        matrix = vectorizer.fit_transform(corpus)
    except Exception:
        st.warning("Erro ao criar índice de busca (dados insuficientes).")
        matrix = None

    q = st.text_input("Digite sua pergunta ou termo (ex: 'barulho à noite', 'uso da piscina por visitantes')")
    top_k = st.slider("Número de resultados", min_value=3, max_value=12, value=6)

    if q and matrix is not None:
        idxs, sims = semantic_search_corpus(vectorizer, matrix, q, top_k=top_k)
        st.markdown(f"### Resultados para: **{q}**")
        if len(idxs) == 0:
            st.info("Nenhum resultado encontrado.")
        for rank, (i, s) in enumerate(zip(idxs, sims), start=1):
            try:
                row = df.iloc[int(i)]
            except Exception:
                continue
            art = f"{row.get('Artigo','')}{(' ' + str(row.get('Item','')) ) if str(row.get('Item','') ) not in ['','nan','None'] else ''}"
            snippet = row.get('Texto_Regra','')
            highlighted = highlight_terms(snippet, q)
            st.markdown(
                f"""
                <div class="result-card">
                  <div style="font-weight:700; font-size:16px; margin-bottom:6px;">{rank}. {art} — Similaridade: {s:.3f}</div>
                  <div style="line-height:1.5;">{highlighted}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption(f"Portal de Convivência {CONDOMINIO_NOME} • Desenvolvido para gestão e boa convivência • Dados: regras_classificadas.csv")
