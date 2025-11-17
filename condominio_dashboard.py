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
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
CONDOMINIO_NOME = "Voraus I"
DATA_CSV = "regras_classificadas.csv"  # output do seu pipeline NLP
ADMIN_PASSWORD = os.environ.get("COND_ADMIN_PWD", "senha123")  # troque em prod

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
    # Normalize column names if necessary
    rename_map = {}
    if 'Categoria_NLP' in df.columns and 'Assunto Principal' not in df.columns:
        rename_map['Categoria_NLP'] = 'Assunto Principal'
    if 'Categoria_Final' in df.columns and 'Assunto Principal' not in df.columns:
        rename_map['Categoria_Final'] = 'Assunto Principal'
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    # Ensure columns exist
    for c in ['Artigo','Item','Texto_Regra','Texto_Limpo','Assunto Principal','Tem_Multa']:
        if c not in df.columns:
            df[c] = ""
    # Friendly penalty label
    df['Penalidade'] = df['Tem_Multa'].astype(str).str.upper().replace({'SIM':'Multa / Advertência','NÃO':'Sem Penalidade'})
    return df

def download_link_df(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False, encoding='utf-8')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download CSV</a>'
    return href

def highlight_terms(text, query):
    if not query:
        return text
    # escape regex and preserve accents by comparing normalized forms
    def norm(s): return unicodedata.normalize('NFD', s).encode('ascii','ignore').decode('utf-8').lower()
    nq = norm(query)
    # naive split by whitespace to highlight each token
    tokens = [t for t in re.split(r'\s+', query.strip()) if t]
    out = text
    for t in tokens:
        if not t: continue
        # find case-insensitive occurrences
        pattern = re.compile(re.escape(t), flags=re.IGNORECASE)
        out = pattern.sub(lambda m: f"<mark style='background: #fff176'>{m.group(0)}</mark>", out)
    return out

def semantic_search_corpus(tfidf_vectorizer, matrix, query, top_k=6):
    q = query if isinstance(query,str) else ""
    q = unicodedata.normalize("NFD", q).encode("ascii","ignore").decode("utf-8").lower()
    q_vec = tfidf_vectorizer.transform([q])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return top_idx, sims[top_idx]

# -----------------------
# LOAD DATA
# -----------------------
df = load_data()

# -----------------------
# SIDEBAR: GLOBAL CONTROLS
# -----------------------
st.sidebar.title("🔧 Controles")
page = st.sidebar.radio("Navegar", ["Dashboard", "Mapa de Regras", "Estatísticas", "Download Center", "Painel do Síndico", "Busca Semântica"])

st.sidebar.markdown("---")
st.sidebar.markdown("Versão: 1.0 • Portal Profissional")

# -----------------------
# HEADER + WELCOME POPUP (simple)
# -----------------------
if "popup_shown" not in st.session_state:
    st.session_state.popup_shown = False

if not st.session_state.popup_shown:
    # popup block
    st.markdown("""
    <style>
    .popup {background:#111827; color:white; padding:26px; border-radius:12px; border:2px solid #818cf8;}
    .popup .row { display:flex; gap:20px; align-items:center; }
    .popup img { width:160px; border-radius:8px; }
    .popup .btn { background:#818cf8; color:#1e1e2d; padding:10px 18px; border-radius:10px; font-weight:700; border:none; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='popup'>
      <div class='row'>
        <img src='https://images.pexels.com/photos/4247766/pexels-photo-4247766.jpeg' alt='convivencia'/>
        <div>
          <h2>🤝 Bem-vindo ao Portal de Convivência - {CONDOMINIO_NOME}</h2>
          <p style='color:#d1d5db'>A informação correta reduz conflitos. Use este portal para localizar regras, verificar penalidades e tomar decisões com base no regulamento.</p>
          <p style='font-style:italic;color:#c7d2fe'>“Boa informação: a chave da boa convivência.”</p>
          <div><button class='btn' onclick="window.dispatchEvent(new Event('closePopup'))">Entrar</button></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # small JS listener to set a flag when clicked
    st.markdown("""
    <script>
    const closePopup = () => document.querySelector('.popup').remove();
    window.addEventListener('closePopup', closePopup);
    </script>
    """, unsafe_allow_html=True)
    st.session_state.popup_shown = True

# -----------------------
# DASHBOARD
# -----------------------
if page == "Dashboard":
    st.title(f"📊 Dashboard - {CONDOMINIO_NOME}")
    if df.empty:
        st.warning("Arquivo CSV não encontrado. Gere 'regras_classificadas.csv' com o pipeline e reabra a página.")
        st.stop()

    # KPI cards
    total_rules = len(df)
    total_articles = df['Artigo'].nunique()
    total_multas = (df['Tem_Multa'].astype(str).str.upper() == 'SIM').sum()
    total_categories = df['Assunto Principal'].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total de Regras", total_rules)
    k2.metric("Artigos (caps)", total_articles)
    k3.metric("Regras com Penalidade", total_multas)
    k4.metric("Categorias (IA)", total_categories)

    st.markdown("---")

    # Pie chart penalties
    fig_pie = px.pie(
        names=['Com Penalidade','Sem Penalidade'],
        values=[total_multas, total_rules - total_multas],
        color_discrete_sequence=['#EF553B','#00CC96']
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Distribuição por Categoria (Top 15)")
    cat_counts = df['Assunto Principal'].value_counts().nlargest(15).reset_index()
    cat_counts.columns = ['Categoria','Quantidade']
    fig_bar = px.bar(cat_counts, x='Quantidade', y='Categoria', orientation='h', color='Quantidade', color_continuous_scale='Blues')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.write("Visual resumo das palavras mais frequentes (sem stopwords):")
    # generate top terms from Texto_Limpo
    all_text = " ".join(df['Texto_Limpo'].fillna("").astype(str).tolist())
    vector = TfidfVectorizer(max_features=100, stop_words='english')  # quick top terms (we assume cleaned)
    v = vector.fit_transform([all_text])
    terms = vector.get_feature_names_out()
    # fallback: just show first N terms
    top_terms = terms[:20].tolist()
    st.write(", ".join(top_terms))

# -----------------------
# MAPA DE REGRAS (Cards + Accordions + Highlight)
# -----------------------
elif page == "Mapa de Regras":
    st.title("📚 Mapa de Regras")

    if df.empty:
        st.warning("CSV ausente.")
        st.stop()

    # Filters
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

    # group by category and show collapsible sections
    cats = df_map['Assunto Principal'].fillna("Sem Categoria").unique()
    for c in sorted(cats):
        sub = df_map[df_map['Assunto Principal'] == c]
        with st.expander(f"📂 {c} — {len(sub)} regras", expanded=False):
            # show as list of accordions for each rule
            for idx, row in sub.iterrows():
                art = f"{row.get('Artigo','')}{(' ' + row.get('Item','')) if row.get('Item','') else ''}"
                texto = row.get('Texto_Regra','')
                # highlight search terms
                highlighted = highlight_terms(texto, search_q) if search_q else texto
                st.markdown(f"**{art}** — {row.get('Penalidade','')}", unsafe_allow_html=True)
                st.markdown(f"<div style='padding:8px;background:#0b1220;color:#dbeafe;border-radius:6px'>{highlighted}</div>", unsafe_allow_html=True)
                st.markdown("---")

# -----------------------
# ESTATÍSTICAS
# -----------------------
elif page == "Estatísticas":
    st.title("📈 Estatísticas do Regulamento")
    if df.empty:
        st.warning("CSV ausente.")
        st.stop()

    # Category distribution
    cat_counts = df['Assunto Principal'].value_counts().reset_index()
    cat_counts.columns = ['Categoria','Quantidade']
    fig = px.pie(cat_counts, names='Categoria', values='Quantidade', title="Distribuição por Categoria (IA)")
    st.plotly_chart(fig, use_container_width=True)

    # Penalidade vs Categoria
    penal = df.groupby(['Assunto Principal','Penalidade']).size().reset_index(name='n')
    fig2 = px.bar(penal, x='n', y='Assunto Principal', color='Penalidade', orientation='h', barmode='stack', title="Penalidades por Categoria")
    st.plotly_chart(fig2, use_container_width=True)

    # Show top terms per cluster/category using TF-IDF
    st.markdown("### Termos mais frequentes por Categoria (Top 6)")
    vectorizer = TfidfVectorizer(max_features=400, stop_words='english', ngram_range=(1,2))
    # create one document per category concatenating Texto_Limpo
    docs = df.groupby('Assunto Principal')['Texto_Limpo'].apply(lambda x: " ".join(x.fillna("").astype(str))).reset_index()
    X = vectorizer.fit_transform(docs['Texto_Limpo'])
    terms = vectorizer.get_feature_names_out()
    for i, row in docs.iterrows():
        cat = row['Assunto Principal']
        vec = X[i].toarray().flatten()
        top_idx = np.argsort(-vec)[:6]
        top_words = [terms[j] for j in top_idx]
        st.write(f"**{cat}**: {', '.join(top_words)}")

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
    st.markdown("Gerar PDF simples de regras filtradas (apenas texto).")
    st.write("Selecione categorias ou use busca para gerar um PDF/texto pronto para impressão.")

    # filtering options
    sel_cat = st.multiselect("Categorias", options=sorted(df['Assunto Principal'].unique()))
    sel_pen = st.multiselect("Penalidade", options=sorted(df['Penalidade'].unique()))
    q = st.text_input("Busca para PDF (opcional)")

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
# PAINEL DO SÍNDICO (edição manual)
# -----------------------
elif page == "Painel do Síndico":
    st.title("🔧 Painel do Síndico (Edição de Categorias)")
    if df.empty:
        st.warning("CSV ausente.")
        st.stop()

    pwd = st.text_input("Senha do Síndico (necessária para salvar alterações)", type="password")
    if pwd != ADMIN_PASSWORD:
        st.warning("Insira a senha correta para editar.")
        st.stop()

    st.success("Autenticado — você pode editar os rótulos das categorias e salvar um novo CSV.")
    # choose a rule
    idx = st.number_input("Escolha índice da regra para editar (linha do CSV)", min_value=0, max_value=len(df)-1, value=0)
    row = df.loc[int(idx)].copy()
    st.markdown("**Regra atual**")
    st.write(row[['Artigo','Item','Texto_Regra','Assunto Principal','Penalidade']])
    new_cat = st.text_input("Novo Assunto Principal (IA)", value=row['Assunto Principal'])
    new_pen = st.selectbox("Penalidade", options=['Multa / Advertência','Sem Penalidade'], index=0 if row['Penalidade']=='Multa / Advertência' else 1)
    if st.button("Salvar alteração"):
        df.at[int(idx),'Assunto Principal'] = new_cat
        df.at[int(idx),'Penalidade'] = new_pen
        # persist to local file with backup
        backup = DATA_CSV.replace(".csv", f".backup")
        if os.path.exists(DATA_CSV):
            os.replace(DATA_CSV, backup)
        df.to_csv(DATA_CSV, index=False, encoding='utf-8')
        st.success("Alteração salva no arquivo CSV e backup criado.")
        st.experimental_rerun()

# -----------------------
# BUSCA SEMÂNTICA
# -----------------------
elif page == "Busca Semântica":
    st.title("🔎 Busca Semântica (local, rápido)")
    if df.empty:
        st.warning("CSV ausente.")
        st.stop()

    # prepare TF-IDF on Texto_Limpo
    corpus = df['Texto_Limpo'].fillna("").astype(str).apply(lambda s: unicodedata.normalize("NFD", s).encode("ascii","ignore").decode("utf-8").lower())
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    matrix = vectorizer.fit_transform(corpus)

    q = st.text_input("Digite sua pergunta ou termo (ex: 'barulho à noite', 'uso da piscina por visitantes')")

    top_k = st.slider("Número de resultados", min_value=3, max_value=12, value=6)
    if q:
        idxs, sims = semantic_search_corpus(vectorizer, matrix, q, top_k=top_k)
        st.markdown(f"Resultados para: **{q}**")
        for rank, (i, s) in enumerate(zip(idxs, sims), start=1):
            row = df.iloc[i]
            art = f"{row['Artigo']}{(' ' + str(row['Item'])) if row['Item'] else ''}"
            snippet = row['Texto_Regra']
            highlighted = highlight_terms(snippet, q)
            st.markdown(f"**{rank}. {art}** — Similaridade: {s:.3f}")
            st.markdown(f"<div style='padding:8px;background:#f3f4f6;border-radius:6px'>{highlighted}</div>", unsafe_allow_html=True)
            st.markdown("---")

# -----------------------
# final footer
# -----------------------
st.markdown("---")
st.caption(f"Portal de Convivência {CONDOMINIO_NOME} • Desenvolvido para gestão e boa convivência • Dados: regras_classificadas.csv")
