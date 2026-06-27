import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
from gensim.models.phrases import Phrases, Phraser
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Temporal Bibliometric LDA", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- UNDUH DATA NLTK ---
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

# --- CSS TEMA MODERN ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Background Body & Header */
    .stApp {
        background-color: #f4f7f6;
    }
    
    .dash-header {
        color: #1a202c;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
        border-bottom: 3px solid #3182ce;
        display: inline-block;
        padding-bottom: 5px;
    }
    
    /* Sidebar Gradient */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2b6cb0 0%, #2c5282 100%);
        border-right: none;
    }
    
    /* Card Dasar Khas Modern */
    .modern-card {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        color: #2b6cb0;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI PEMROSESAN (CACHED) ---

@st.cache_data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Abstract']).copy()
    
    predatory_keywords = [
        'omics', 'waset', 'academic journals', 'sciedu', 
        'david publishing', 'baishideng', 'bentham open', 
        'iosr', 'science domains', 'lap lambert'
    ]
    df['Publisher_lower'] = df['Publisher'].astype(str).str.lower()
    df['Source_lower'] = df['Source'].astype(str).str.lower()
    
    def is_predatory(row):
        for keyword in predatory_keywords:
            if keyword in row['Publisher_lower'] or keyword in row['Source_lower']:
                return True
        return False
        
    df['is_predatory'] = df.apply(is_predatory, axis=1)
    df_predatory = df[df['is_predatory']].copy()
    num_predatory = len(df_predatory)
    df_clean = df[~df['is_predatory']].copy()
    
    df_clean = df_clean.drop(columns=['Publisher_lower', 'Source_lower', 'is_predatory'])
    df_predatory = df_predatory.drop(columns=['Publisher_lower', 'Source_lower', 'is_predatory'])
    
    return df, df_clean, df_predatory, num_predatory

@st.cache_data(show_spinner="Tokenisasi & Pembentukan N-Gram (Bigram)...")
def build_nlp_pipeline(df_clean):
    stop_words = set(stopwords.words('english'))
    custom_stops = {'research', 'paper', 'method', 'using', 'result', 'data', 'science', 'model', 'approach', 'study', 'show', 'propose'}
    stop_words.update(custom_stops)

    def preprocess_basic(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return tokens

    docs = df_clean['Abstract'].apply(preprocess_basic).tolist()

    # Bigram & Trigram (Phrase ID)
    bigram = Phrases(docs, min_count=3, threshold=10)
    bigram_mod = Phraser(bigram)
    
    docs_bigram = [bigram_mod[doc] for doc in docs]
    trigram = Phrases(docs_bigram, min_count=3, threshold=10)
    trigram_mod = Phraser(trigram)
    
    clean_docs = [trigram_mod[doc] for doc in docs_bigram]
    df_clean['clean_tokens'] = clean_docs
    
    # Dictionary & Corpus
    dictionary = corpora.Dictionary(clean_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus_tf = [dictionary.doc2bow(text) for text in clean_docs]
    
    # TF-IDF
    tfidf_model = TfidfModel(corpus_tf)
    corpus_tfidf = tfidf_model[corpus_tf]
    
    return df_clean, dictionary, corpus_tf, corpus_tfidf, clean_docs

@st.cache_resource(show_spinner="Melatih Model LDA Global...")
def train_lda_model(_corpus_tfidf, _dictionary, k):
    # Latih model tunggal (Global Model)
    lda_model = LdaModel(corpus=_corpus_tfidf, id2word=_dictionary, num_topics=k,
                         random_state=42, passes=5, iterations=100, alpha='auto')
    return lda_model

# --- SIDEBAR INTERAKTIF ---
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-top: 1rem; margin-bottom: 2rem;'>
        <div style='color: white; font-weight: 800; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 1px;'>Temporal LDA</div>
        <div style='color: #bee3f8; font-size: 0.85rem; font-weight: 600;'>Bibliometric Evolution Engine</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='color: white; font-weight: 700; margin-bottom: 0.5rem; padding-left: 1rem;'>📁 Data Sumber</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Format harus CSV dan memiliki kolom 'Abstract'.")
    use_default = st.checkbox("Gunakan Dataset Bawaan", value=False)
    
    st.markdown("<hr style='border-top: 1px solid rgba(255, 255, 255, 0.15); margin: 1.5rem 1rem 1.5rem 1rem;'>", unsafe_allow_html=True)
    
    selected_tab = option_menu(
        menu_title=None,
        options=["Dashboard", "Topografi Digital", "Distribusi Kata", "Jalur Migrasi Riset"],
        icons=["house-fill", "map", "cloud-fill", "graph-up"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "rgba(255,255,255,0.9)", "font-size": "16px", "margin-left": "10px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "color": "rgba(255,255,255,0.8)", "font-weight": "600", "padding": "0.8rem 1rem"},
            "nav-link-selected": {"background-color": "rgba(255,255,255,0.15)", "color": "#fff", "font-weight": "800", "border-radius": "8px"},
        }
    )
    
    st.markdown("<hr style='border-top: 1px solid rgba(255, 255, 255, 0.15); margin: 2rem 1rem 1.5rem 1rem;'>", unsafe_allow_html=True)
    
    st.markdown("<div style='color: white; font-weight: 700; margin-bottom: 0.5rem; padding-left: 1rem;'>⚙️ Parameter Model</div>", unsafe_allow_html=True)
    optimal_k = st.slider("Jumlah Topik (K)", min_value=2, max_value=20, value=16, step=1, 
                          help="Geser untuk mengatur jumlah topik laten yang akan diekstraksi.")
    
    st.markdown("""
    <div style='padding: 1rem; color: #bee3f8; font-size: 0.8rem; line-height: 1.4;'>
        <strong>Metode:</strong> Global LDA Model<br>
        <strong>Fitur:</strong> TF-IDF + Phrase ID (Bigram)<br>
        Model akan dilatih ulang secara otomatis setiap kali Anda mengubah (K).
    </div>
    """, unsafe_allow_html=True)


# --- INISIALISASI DATA UTAMA ---
data_source = None
if uploaded_file is not None:
    data_source = uploaded_file
elif use_default:
    data_source = "dataset_riset_Data Science-Semantic Scholer.csv"

if data_source is None:
    st.markdown("""
    <div style='padding: 3rem; text-align: center; background-color: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 2rem;'>
        <h2 style='color: #2b6cb0; margin-bottom: 1rem;'>👋 Selamat Datang di Temporal LDA Engine</h2>
        <p style='color: #4a5568; font-size: 1.1rem;'>Silakan unggah dataset berformat CSV melalui panel <b>Sidebar di sebelah kiri</b> untuk memulai analisis topik.<br><br>Atau Anda dapat mencentang opsi <b>'Gunakan Dataset Bawaan'</b> untuk melihat demonstrasi.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    df_raw, df_clean, df_predatory, num_predatory = load_and_clean_data(data_source)
    df_clean, dictionary, corpus_tf, corpus_tfidf, clean_docs = build_nlp_pipeline(df_clean)
except Exception as e:
    st.error(f"Gagal memuat atau memproses dataset. Pastikan dataset CSV Anda memiliki struktur yang sesuai.\\n\\nError Detail: {e}")
    st.stop()

# --- MELATIH MODEL BERDASARKAN K SLIDER ---
lda_model = train_lda_model(corpus_tfidf, dictionary, optimal_k)


# --- AREA KONTEN UTAMA ---
st.markdown(f"<div class='dash-header'>{selected_tab}</div>", unsafe_allow_html=True)

if selected_tab == "Dashboard":
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='modern-card'><div style='color:#718096; font-size:0.8rem; font-weight:700; text-transform:uppercase;'>Total Dokumen</div><div style='font-size:1.8rem; font-weight:800; color:#2d3748;'>{len(df_raw):,}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='modern-card'><div style='color:#38a169; font-size:0.8rem; font-weight:700; text-transform:uppercase;'>Dokumen Dianalisis</div><div style='font-size:1.8rem; font-weight:800; color:#276749;'>{len(df_clean):,}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='modern-card'><div style='color:#e53e3e; font-size:0.8rem; font-weight:700; text-transform:uppercase;'>Predator Dihapus</div><div style='font-size:1.8rem; font-weight:800; color:#9b2c2c;'>{num_predatory:,}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='modern-card'><div style='color:#3182ce; font-size:0.8rem; font-weight:700; text-transform:uppercase;'>Topik Laten (K)</div><div style='font-size:1.8rem; font-weight:800; color:#2b6cb0;'>{optimal_k}</div></div>", unsafe_allow_html=True)

    # Tables
    st.markdown("<div class='modern-card'><div class='card-title'>Sampel Data Publikasi Bersih</div>", unsafe_allow_html=True)
    st.dataframe(df_clean[['Title', 'Year', 'Source', 'Abstract']].head(100), use_container_width=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_tab == "Topografi Digital":
    st.markdown("""
    <div class='modern-card'>
        <div class='card-title'>Interaksi & Jarak Semantik Antar Topik (pyLDAvis)</div>
        <p style='color: #4a5568; font-size: 0.9rem;'>
            Gelembung mewakili prevalensi topik dalam korpus secara global. Jarak antar gelembung mendefinisikan seberapa mirip/berbedanya topik tersebut secara semantik.
        </p>
    """, unsafe_allow_html=True)
    
    with st.spinner("Memuat visualisasi interaktif pyLDAvis..."):
        # pyLDAvis lebih representatif jika dipasangkan dengan Term Frequency murni (corpus_tf)
        vis = gensimvis.prepare(lda_model, corpus_tf, dictionary)
        html_string = pyLDAvis.prepared_data_to_html(vis)
        components.html(html_string, width=1300, height=850, scrolling=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_tab == "Distribusi Kata":
    topics = lda_model.show_topics(num_topics=optimal_k, num_words=15, formatted=False)
    
    st.markdown("<div class='modern-card'><div class='card-title'>Awan Kata Kunci (Word Cloud)</div>", unsafe_allow_html=True)
    cols = 4
    rows = (len(topics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    fig.patch.set_facecolor('none') 
    axes = np.array(axes).flatten()
    
    for i in range(len(axes)):
        if i < len(topics):
            topic_words = dict(topics[i][1])
            cloud = WordCloud(background_color='white', width=400, height=400, colormap='GnBu_r').generate_from_frequencies(topic_words)
            axes[i].imshow(cloud, interpolation='bilinear')
            axes[i].set_title(f'Topik {topics[i][0]}', fontdict=dict(size=18, color='#2b6cb0', weight='bold'))
        axes[i].axis('off')
    
    st.pyplot(fig, transparent=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='modern-card'><div class='card-title'>Probabilitas Istilah (Bar Chart)</div>", unsafe_allow_html=True)
    fig2, axes2 = plt.subplots(rows, cols, figsize=(20, 5*rows))
    fig2.patch.set_facecolor('none')
    axes2 = np.array(axes2).flatten()
    
    for i in range(len(axes2)):
        if i < len(topics):
            words = [word for word, prob in topics[i][1]]
            probs = [prob for word, prob in topics[i][1]]
            axes2[i].barh(words, probs, color='#3182ce', edgecolor='none', alpha=0.9)
            axes2[i].set_title(f'Topik {topics[i][0]}', fontdict=dict(color='#2d3748', weight='bold'))
            axes2[i].invert_yaxis()
            axes2[i].spines['top'].set_visible(False)
            axes2[i].spines['right'].set_visible(False)
            axes2[i].spines['bottom'].set_color('#e2e8f0')
            axes2[i].spines['left'].set_color('#e2e8f0')
        else:
            axes2[i].axis('off')
            
    st.pyplot(fig2, transparent=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_tab == "Jalur Migrasi Riset":
    @st.cache_data
    def get_temporal_data(_df, _model, _corpus):
        def get_dominant_topic(bow):
            topic_probs = _model.get_document_topics(bow)
            if not topic_probs: return None
            return max(topic_probs, key=lambda x: x[1])[0]

        _df['dominant_topic'] = [get_dominant_topic(bow) for bow in _corpus]
        temporal_topics = _df.groupby(['Year', 'dominant_topic']).size().unstack(fill_value=0)
        # Filter mulai dari 2017 ke atas
        temporal_topics = temporal_topics[temporal_topics.index >= 2017]
        temporal_perc = temporal_topics.div(temporal_topics.sum(axis=1), axis=0) * 100
        
        temporal_counts = _df.groupby(['Year', 'dominant_topic']).size().reset_index(name='Count')
        temporal_counts = temporal_counts[temporal_counts['Year'] >= 2017].sort_values(by=['Year', 'dominant_topic'])
        return temporal_perc, temporal_counts
        
    with st.spinner("Menghitung data tren evolusi temporal..."):
        temporal_perc, temporal_counts = get_temporal_data(df_clean.copy(), lda_model, corpus_tfidf)
    
    st.markdown("""
    <div class='modern-card'>
        <div class='card-title'>Menyingkap Evolusi (Tren Topik 2017 - Sekarang)</div>
        <p style='color: #4a5568; font-size: 0.9rem;'>
            Garis-garis ini merepresentasikan "Siklus Kehidupan Riset" secara global. Identifikasi tren mana yang abadi (SOTA), musiman (Booming), atau meredup (Sunset).
        </p>
    """, unsafe_allow_html=True)
    
    fig_line, ax_line = plt.subplots(figsize=(14, 6))
    fig_line.patch.set_facecolor('none')
    
    temporal_perc.plot(kind='line', marker='o', ax=ax_line, linewidth=3, markersize=8, colormap='tab20')
    ax_line.set_ylabel('Proporsi Ketertarikan (%)', weight='bold', labelpad=10, color='#2d3748', fontsize=12)
    ax_line.set_xlabel('Tahun', weight='bold', labelpad=10, color='#2d3748', fontsize=12)
    ax_line.grid(True, linestyle='--', alpha=0.5, color='#cbd5e0')
    ax_line.set_xticks(temporal_perc.index)
    
    ax_line.spines['top'].set_visible(False)
    ax_line.spines['right'].set_visible(False)
    ax_line.spines['bottom'].set_color('#cbd5e0')
    ax_line.spines['left'].set_color('#cbd5e0')
    ax_line.legend(title='ID Topik', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    st.pyplot(fig_line, transparent=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='modern-card'><div class='card-title'>Peta Panas Kedekatan Semantik per Tahun</div>", unsafe_allow_html=True)
    fig_heat, ax_heat = plt.subplots(figsize=(14, 6))
    fig_heat.patch.set_facecolor('none')
    sns.heatmap(temporal_perc, annot=True, cmap='Blues', fmt='.1f', ax=ax_heat, linewidths=.5, cbar_kws={'label': 'Proporsi (%)'})
    ax_heat.set_xlabel('ID Topik', weight='bold', labelpad=10, color='#2d3748')
    ax_heat.set_ylabel('Tahun', weight='bold', labelpad=10, color='#2d3748')
    st.pyplot(fig_heat, transparent=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- SANKEY DIAGRAM ---
    st.markdown("<div class='modern-card'><div class='card-title'>Jalur Migrasi Topik (Sankey Diagram)</div>", unsafe_allow_html=True)

    
    years = sorted(temporal_counts['Year'].unique().tolist())
    topics_id = sorted(temporal_counts['dominant_topic'].unique().tolist())
    
    labels = [str(y) for y in years] + [f"Topik {t}" for t in topics_id]
    label_map = {name: i for i, name in enumerate(labels)}
    
    source = [label_map[str(row['Year'])] for _, row in temporal_counts.iterrows()]
    target = [label_map[f"Topik {row['dominant_topic']}"] for _, row in temporal_counts.iterrows()]
    value = [row['Count'] for _, row in temporal_counts.iterrows()]
        
    fig_sankey = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = "#3182ce"
        ),
        link = dict(
          source = source,
          target = target,
          value = value,
          color = "rgba(160, 174, 192, 0.4)"
        )
    )])
    fig_sankey.update_layout(title_text="Aliran Proporsi Publikasi dari Tahun ke Topik Laten", font_size=12, height=550, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
