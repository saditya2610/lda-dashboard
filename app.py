import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Analisis Riset Data Science", page_icon="SB", layout="wide", initial_sidebar_state="expanded")

# --- CSS TEMA SB ADMIN 2 ---
st.markdown("""
<style>
    /* Font Nunito khas SB Admin 2 */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@200;300;400;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Nunito', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    
    /* Sembunyikan elemen Streamlit, tapi biarkan tombol sidebar collapse tetap ada */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Background Body SB Admin 2 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        background-color: #f8f9fc;
    }
    .stApp {
        background-color: #f8f9fc;
    }
    
    /* Judul Halaman SB Admin 2 */
    .dash-header {
        color: #5a5c69;
        font-size: 1.75rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Gradient khas SB Admin 2 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4e73df 10%, #224abe 100%);
        border-right: none;
    }
    
    /* Card Dasar Khas SB Admin 2 */
    .sb-card {
        background-color: #fff;
        background-clip: border-box;
        border: 1px solid #e3e6f0;
        border-radius: 0.35rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin-bottom: 1.5rem;
    }
    .sb-card-header {
        padding: .75rem 1.25rem;
        margin-bottom: 0;
        background-color: #f8f9fc;
        border-bottom: 1px solid #e3e6f0;
        color: #4e73df;
        font-weight: 700;
        border-top-left-radius: 0.35rem;
        border-top-right-radius: 0.35rem;
    }
    .sb-card-body {
        padding: 1.25rem;
    }
    
    /* Tabel tanpa border luar agar pas di card */
    [data-testid="stDataFrame"] {
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- UNDUH DATA NLTK ---
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

# --- FUNGSI PEMROSESAN ---
@st.cache_data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['periode'] = df['Year'].apply(lambda x: 'Periode_1' if x < 2022 else 'Periode_2')
    
    predatory_keywords = [
        'omics', 'waset', 'academic journals', 'sciedu', 
        'david publishing', 'baishideng', 'bentham open', 
        'iosr', 'science domains', 'lap lambert'
    ]
    df['Publisher_lower'] = df['Publisher'].astype(str).str.lower().fillna('')
    df['Source_lower'] = df['Source'].astype(str).str.lower().fillna('')
    
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

@st.cache_data
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    custom_stops = {'research', 'paper', 'method', 'using', 'result', 'data', 'science', 'also', 'can', 'study', 'model', 'based'}
    stop_words.update(custom_stops)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

@st.cache_data(show_spinner=False)
def preprocess_dataframe(df_clean):
    df_clean['clean_tokens'] = df_clean['Abstract'].apply(preprocess_text)
    return df_clean

@st.cache_resource(show_spinner=False)
def train_lda_models(df_p1, df_p2, num_topics=5):
    def prepare_lda_input(tokens_list):
        dictionary = corpora.Dictionary(tokens_list)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in tokens_list]
        return dictionary, corpus

    dict_p1, corpus_p1 = prepare_lda_input(df_p1['clean_tokens'])
    dict_p2, corpus_p2 = prepare_lda_input(df_p2['clean_tokens'])

    lda_model_p1 = LdaModel(corpus=corpus_p1, id2word=dict_p1, num_topics=num_topics,
                            random_state=42, passes=2, alpha='auto')
    lda_model_p2 = LdaModel(corpus=corpus_p2, id2word=dict_p2, num_topics=num_topics,
                            random_state=42, passes=2, alpha='auto')
                            
    return (dict_p1, corpus_p1, lda_model_p1), (dict_p2, corpus_p2, lda_model_p2)


# --- INISIALISASI DATA ---
filepath = "dataset_riset_Data Science-Semantic Scholer.csv"
try:
    df_raw, df_clean, df_predatory, num_predatory = load_and_clean_data(filepath)
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()


# --- SIDEBAR SB ADMIN 2 ---
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-top: 1rem; margin-bottom: 2rem;'>
        <div style='color: white; font-weight: 800; font-size: 1.1rem; text-transform: uppercase; line-height: 1.3;'>Temporal Bibliometric<br>Topic Modeling</div>
    </div>
    <hr style='border-top: 1px solid rgba(255, 255, 255, 0.15); margin: 0 1rem 1rem 1rem;'>
    """, unsafe_allow_html=True)
    
    selected_tab = option_menu(
        menu_title=None,
        options=["Dashboard", "Visualisasi Topik", "Distribusi Kata", "Tren Temporal"],
        icons=["tachometer", "diagram-3", "cloud", "graph-up"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "rgba(255,255,255,.8)", "font-size": "15px", "margin-left": "10px"}, 
            "nav-link": {"font-size": "13px", "text-align": "left", "margin":"0px", "color": "rgba(255,255,255,.8)", "font-weight": "700", "padding": "1rem"},
            "nav-link-selected": {"background-color": "rgba(255,255,255,0.15)", "color": "#fff", "font-weight": "800", "border-radius": "0.35rem"},
        }
    )


# --- PEMROSESAN LATAR BELAKANG ---
if 'df_processed' not in st.session_state:
    with st.spinner("Menginisialisasi model dan memproses NLP (Proses pertama memakan waktu sejenak)..."):
        df_processed = preprocess_dataframe(df_clean.copy())
        df_p1 = df_processed[df_processed['periode'] == 'Periode_1']
        df_p2 = df_processed[df_processed['periode'] == 'Periode_2']
        
        models_p1, models_p2 = train_lda_models(df_p1, df_p2, num_topics=5)
        
        st.session_state['df_processed'] = df_processed
        st.session_state['models_p1'] = models_p1
        st.session_state['models_p2'] = models_p2
else:
    df_processed = st.session_state['df_processed']
    models_p1 = st.session_state['models_p1']
    models_p2 = st.session_state['models_p2']

dict_p1, corpus_p1, lda_model_p1 = models_p1
dict_p2, corpus_p2, lda_model_p2 = models_p2


# --- AREA KONTEN UTAMA ---
st.markdown("<div class='dash-header'>Dashboard Analisis Topik</div>", unsafe_allow_html=True)

if selected_tab == "Dashboard":
    # --- METRICS CARDS (SB Admin 2 Style) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="border-left: .25rem solid #4e73df!important; border-radius: .35rem; padding: 1.25rem; background-color: #fff; box-shadow: 0 .15rem 1.75rem 0 rgba(58,59,69,.15); margin-bottom: 1.5rem;">
            <div style="font-size: .7rem; font-weight: 700; color: #4e73df; text-transform: uppercase; margin-bottom: 5px;">Total Dokumen Awal</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #5a5c69;">{len(df_raw):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div style="border-left: .25rem solid #1cc88a!important; border-radius: .35rem; padding: 1.25rem; background-color: #fff; box-shadow: 0 .15rem 1.75rem 0 rgba(58,59,69,.15); margin-bottom: 1.5rem;">
            <div style="font-size: .7rem; font-weight: 700; color: #1cc88a; text-transform: uppercase; margin-bottom: 5px;">Dokumen Bersih (Dianalisis)</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #5a5c69;">{len(df_clean):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div style="border-left: .25rem solid #e74a3b!important; border-radius: .35rem; padding: 1.25rem; background-color: #fff; box-shadow: 0 .15rem 1.75rem 0 rgba(58,59,69,.15); margin-bottom: 1.5rem;">
            <div style="font-size: .7rem; font-weight: 700; color: #e74a3b; text-transform: uppercase; margin-bottom: 5px;">Jurnal Predator (Dihapus)</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #5a5c69;">{num_predatory:,}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- TABLES CARDS ---
    st.markdown("""
    <div class="sb-card">
        <div class="sb-card-header">Tabel Sampel Data Bersih</div>
        <div class="sb-card-body">
    """, unsafe_allow_html=True)
    st.dataframe(df_clean[['Title', 'Year', 'Abstract', 'periode']].head(100), use_container_width=True, height=350)
    st.markdown("</div></div>", unsafe_allow_html=True)

    if num_predatory > 0:
        st.markdown("""
        <div class="sb-card" style="border-left: .25rem solid #e74a3b!important;">
            <div class="sb-card-header" style="color: #e74a3b;">Tabel Jurnal Predator Terfilter</div>
            <div class="sb-card-body">
        """, unsafe_allow_html=True)
        st.dataframe(df_predatory[['Title', 'Year', 'Source', 'Publisher', 'Abstract']].head(100), use_container_width=True, height=250)
        st.markdown("</div></div>", unsafe_allow_html=True)

elif selected_tab == "Visualisasi Topik":
    st.markdown("""
    <div class="sb-card">
        <div class="sb-card-header">Klaster Topik Interaktif (pyLDAvis)</div>
        <div class="sb-card-body">
    """, unsafe_allow_html=True)
    
    col_sel, _ = st.columns([1, 4])
    with col_sel:
        period_sel = st.selectbox("Pilih Periode Waktu:", ["Periode 1 (< 2022)", "Periode 2 (>= 2022)"])
    
    st.markdown("<hr style='border-top: 1px solid #e3e6f0; margin-top:1rem; margin-bottom:1rem;'>", unsafe_allow_html=True)
    
    with st.spinner("Memuat visualisasi interaktif..."):
        if period_sel == "Periode 1 (< 2022)":
            vis = gensimvis.prepare(lda_model_p1, corpus_p1, dict_p1)
        else:
            vis = gensimvis.prepare(lda_model_p2, corpus_p2, dict_p2)
            
        html_string = pyLDAvis.prepared_data_to_html(vis)
        components.html(html_string, width=1300, height=800, scrolling=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

elif selected_tab == "Distribusi Kata":
    col_sel, _ = st.columns([1, 4])
    with col_sel:
        period_sel = st.selectbox("Pilih Periode Waktu:", ["Periode 1 (< 2022)", "Periode 2 (>= 2022)"])
    
    model = lda_model_p1 if period_sel == "Periode 1 (< 2022)" else lda_model_p2
    topics = model.show_topics(num_topics=5, formatted=False)
    
    st.markdown("""
    <div class="sb-card">
        <div class="sb-card-header">Awan Kata (Word Cloud)</div>
        <div class="sb-card-body">
    """, unsafe_allow_html=True)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    fig.patch.set_facecolor('none') 
    
    for i, ax in enumerate(axes.flatten()):
        topic_words = dict(topics[i][1])
        # SB Admin 2 Primary color palette
        cloud = WordCloud(background_color='white', width=400, height=400, colormap='Blues_r').generate_from_frequencies(topic_words)
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_title(f'Topik {i}', fontdict=dict(size=18, color='#4e73df', weight='bold'))
        ax.axis('off')
    st.pyplot(fig, transparent=True)
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sb-card">
        <div class="sb-card-header">Grafik Batang Probabilitas Kata</div>
        <div class="sb-card-body">
    """, unsafe_allow_html=True)
    
    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    fig2.patch.set_facecolor('none')
    
    for i, ax in enumerate(axes2.flatten()):
        words = [word for word, prob in topics[i][1]]
        probs = [prob for word, prob in topics[i][1]]
        ax.barh(words, probs, color='#4e73df', edgecolor='none', alpha=0.9) # SB Admin 2 Primary Blue
        ax.set_title(f'Topik {i}', fontdict=dict(color='#5a5c69', weight='bold'))
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#e3e6f0')
        ax.spines['left'].set_color('#e3e6f0')
        
    st.pyplot(fig2, transparent=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

elif selected_tab == "Tren Temporal":
    @st.cache_data
    def get_temporal_data(df):
        def get_dominant_topic(bow, model):
            topic_probs = model.get_document_topics(bow)
            if not topic_probs: return None
            return max(topic_probs, key=lambda x: x[1])[0]

        df['dominant_topic'] = df.apply(lambda row:
            get_dominant_topic(dict_p1.doc2bow(row['clean_tokens']), lda_model_p1) if row['periode'] == 'Periode_1'
            else get_dominant_topic(dict_p2.doc2bow(row['clean_tokens']), lda_model_p2), axis=1)

        temporal_topics = df.groupby(['Year', 'dominant_topic']).size().unstack(fill_value=0)
        temporal_perc = temporal_topics.div(temporal_topics.sum(axis=1), axis=0) * 100
        return temporal_perc
        
    with st.spinner("Menghitung data tren temporal..."):
        temporal_perc = get_temporal_data(df_processed.copy())
    
    st.markdown("""
    <div class="sb-card">
        <div class="sb-card-header">Peta Panas Evolusi Topik (Heatmap)</div>
        <div class="sb-card-body">
    """, unsafe_allow_html=True)
    
    fig_heat, ax_heat = plt.subplots(figsize=(12, 6))
    fig_heat.patch.set_facecolor('none')
    sns.heatmap(temporal_perc, annot=True, cmap='Blues', fmt='.1f', ax=ax_heat, linewidths=.5, cbar_kws={'label': 'Frekuensi Relatif (%)'})
    ax_heat.set_xlabel('ID Topik', weight='bold', labelpad=10, color='#5a5c69')
    ax_heat.set_ylabel('Tahun', weight='bold', labelpad=10, color='#5a5c69')
    st.pyplot(fig_heat, transparent=True)
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sb-card">
        <div class="sb-card-header">Tren Seiring Waktu (Grafik Garis)</div>
        <div class="sb-card-body">
    """, unsafe_allow_html=True)
    
    fig_line, ax_line = plt.subplots(figsize=(12, 6))
    fig_line.patch.set_facecolor('none')
    
    temporal_perc.plot(kind='line', marker='o', ax=ax_line, linewidth=3, markersize=8, colormap='Set1')
    ax_line.set_ylabel('Frekuensi Relatif (%)', weight='bold', labelpad=10, color='#5a5c69')
    ax_line.set_xlabel('Tahun', weight='bold', labelpad=10, color='#5a5c69')
    ax_line.grid(True, linestyle='--', alpha=0.5, color='#e3e6f0')
    
    ax_line.spines['top'].set_visible(False)
    ax_line.spines['right'].set_visible(False)
    ax_line.spines['bottom'].set_color('#e3e6f0')
    ax_line.spines['left'].set_color('#e3e6f0')
    ax_line.legend(title='ID Topik', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    st.pyplot(fig_line, transparent=True)
    st.markdown("</div></div>", unsafe_allow_html=True)
