import re
import pypdf
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

# Garantir stopwords (execute uma vez se necessário)
try:
    stopwords.words('portuguese')
except:
    nltk.download('stopwords')

PDF_FILENAME = "024 - REGULAMENTO INTERNO - reformulado.pdf"
OUTPUT_CSV = "regras_condominio.csv"

PENALTY_KEYWORDS = [
    "multa", "advertência", "penalidade", "sanção", "suspensão",
    "sujeita a", "incide em", "será punido", "carta de advertência"
]

def extract_text(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for p in reader.pages:
        page_text = p.extract_text() or ""
        text += page_text + "\n"
    return text

def split_articles(full_text):
    """
    Retorna lista de (article_label, article_block_text, start_idx, end_idx)
    Detecta marcadores 'Art. 1º', 'Art. 1', 'Art 1º' (case-insensitive).
    """
    # normalize spaces and newlines to single spaces to make indexing reliable
    norm = full_text.replace('\r', ' ').replace('\n', ' ')
    # find all article markers
    art_pat = re.compile(r'(Art\.?\s*\d+º?)', re.IGNORECASE)
    matches = list(art_pat.finditer(norm))
    articles = []
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(norm)
        block = norm[start:end].strip()
        articles.append((label, block))
    return articles

def split_subitems(article_block):
    """
    Dentro do bloco de um artigo, identifica subitens numerados (ex: 3.1, 3.1.2 ...)
    Retorna (article_main_text, list_of_subitems) onde each subitem = (item_label, item_text)
    """
    # normalize: ensure there's spacing around common separators
    block = article_block.strip()
    # pattern to find subitem markers like 3.1 or 12.10 or 3.1.2 (one or more dots)
    # allow optional trailing punctuation (comma, dash, colon) immediately after the number
    sub_pat = re.compile(r'(\d+(?:\.\d+)+)\s*(?:[–—\-:]|,)?\s*', re.IGNORECASE)
    matches = list(sub_pat.finditer(block))
    if not matches:
        # no subitems found
        return block, []

    subitems = []
    # text before first subitem is the article's main text
    first_start = matches[0].start()
    main_text = block[:first_start].strip()

    for i, m in enumerate(matches):
        label = m.group(1).strip()
        content_start = m.end()
        content_end = matches[i+1].start() if i+1 < len(matches) else len(block)
        content = block[content_start:content_end].strip()
        # remove trailing punctuation like a solitary comma at end
        content = content.rstrip(' ,;:')
        subitems.append((label, content))
    return main_text, subitems

def classify_penalty(text):
    low = text.lower()
    return "SIM" if any(k in low for k in PENALTY_KEYWORDS) else "NÃO"

def clean_text_for_nlp(text):
    stop_pt = set(stopwords.words('portuguese'))
    t = text.lower()
    t = re.sub(f"[{re.escape(string.punctuation)}]", "", t)
    t = re.sub(r"\d+", "", t)
    words = [w for w in t.split() if w and w not in stop_pt]
    return " ".join(words)

def parse_pdf_to_rows(pdf_path):
    raw = extract_text(pdf_path)
    articles = split_articles(raw)
    rows = []
    for art_label, block in articles:
        # attempt to split subitems
        main_text, subitems = split_subitems(block)
        parent = art_label
        # if main_text non-empty -> add as its own rule (Article main)
        if main_text and len(main_text.strip())>0:
            row = {
                "Artigo": art_label,
                "Item": "",
                "Texto_Regra": main_text,
                "Parent_Art": parent,
                "Tem_Multa": classify_penalty(main_text),
                "Assunto_Bruto": "A Classificar (NLP)"
            }
            rows.append(row)
        # add each subitem as separate row
        for item_label, item_text in subitems:
            if not item_text:
                continue
            row = {
                "Artigo": art_label,
                "Item": item_label,
                "Texto_Regra": item_text,
                "Parent_Art": parent,
                "Tem_Multa": classify_penalty(item_text),
                "Assunto_Bruto": "A Classificar (NLP)"
            }
            rows.append(row)
    return rows

if __name__ == "__main__":
    rows = parse_pdf_to_rows(PDF_FILENAME)
    df = pd.DataFrame(rows, columns=["Artigo","Item","Texto_Regra","Parent_Art","Tem_Multa","Assunto_Bruto"])
    # add Texto_Limpo for NLP
    df["Texto_Limpo"] = df["Texto_Regra"].apply(clean_text_for_nlp)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Gerado {OUTPUT_CSV} com {len(df)} linhas")
