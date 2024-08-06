import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import re
import plotly.express as px

st.set_page_config(layout='wide', page_title='CosmetriX')

# Veri setlerini yükle ve birleştir
@st.cache_data
def get_data():
    product_info = pd.read_csv('pythonProject3/product_info.csv')
    output_data = pd.read_excel('pythonProject3/output.xlsx')
    df = pd.merge(product_info, output_data, on='product_id', how='left')
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.rename(columns={'product_name_x': 'product_name', 'primary_category_x': 'primary_category',
                            'secondary_category_x': 'secondary_category'})
    return df

veriler = get_data()

# Metin temizleme fonksiyonu (Türkçe karakterleri korur)
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9çğüşıöğü\s]', '', text)  # Türkçe karakterleri korur
        text = re.sub(r'\s+', ' ', text).strip()
    return text

# Metin verilerini temizle ve birleştir
def preprocess_data(df):
    text_columns = ['product_name', 'highlights', 'ingredients']
    for col in text_columns:
        if col in df.columns:
            df[f'clean_{col}'] = df[col].fillna('').apply(clean_text)
    problem_columns = ['problem1', 'problem2', 'problem3']
    df['clean_problems'] = df[problem_columns].fillna('').agg(' '.join, axis=1).apply(clean_text)
    df['all_text'] = df[[f'clean_{col}' for col in text_columns if f'clean_{col}' in df.columns] + ['clean_problems']].fillna('').agg(' '.join, axis=1)
    return df

# TF-IDF vektörizasyonu
def vectorize_data(df):
    tfidf_text = TfidfVectorizer(stop_words='english')
    tfidf_matrix_text = tfidf_text.fit_transform(df['all_text'])
    tfidf_problems = TfidfVectorizer(stop_words='english')
    tfidf_matrix_problems = tfidf_problems.fit_transform(df['clean_problems'])
    return np.hstack([tfidf_matrix_text.toarray(), tfidf_matrix_problems.toarray()])

# Öneri fonksiyonu
def get_recommendations(product_name, df, tfidf_matrix, n=10):
    idx = df.index[df['product_name'] == product_name].tolist()
    if not idx:
        st.write("Ürün adı bulunamadı. Lütfen ürün adını kontrol edin.")
        return None
    idx = idx[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx:idx + 1], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    product_indices = [i[0] for i in sim_scores]
    similar_products = df.iloc[product_indices].copy()

    if 'price_usd' in df.columns:
        original_price = df.loc[idx, 'price_usd']
        similar_products = similar_products[similar_products['price_usd'] <= original_price * 1.3]

    if 'rating' in df.columns:
        similar_products = similar_products[similar_products['rating'] >= 4]

    similar_products['similarity_score'] = cosine_sim[similar_products.index]
    similar_products = similar_products.sort_values('similarity_score', ascending=False)
    top_products = similar_products.head(n)

    result_columns = ['product_name', 'similarity_score']
    for col in ['price_usd', 'rating', 'reviews', 'loves_count', 'category', 'clean_problems']:
        if col in df.columns:
            result_columns.append(col)

    results = top_products[result_columns]
    return results

# Ana sayfa sekmesi


# Başlık ve açıklama
st.markdown("""
    <div style='text-align: center; margin-right: 30px;'>
        <h1 style='color: pink; font-size: 80px;'>CosmetriX</h1>
    </div>
    """, unsafe_allow_html=True)

# Metin ve resim
st.markdown("""
    <div style='display: flex; align-items: center;'>
        <img src='https://lh3.googleusercontent.com/d/1HGbLvTh-Zd0DD5N5AzM4pd82NQRL4KE9=s390' width='300' height='280' style='margin-top: -120px;'>
          <h3 style='font-family: Arial; font-size: 19px; color: #8e44ad; margin-left: 20px; margin-bottom: -20px;'>
            CİLT SAĞLIĞINA YOLCULUK: Verinin gücüyle kişiselleştirilmiş bakım....
        </h3>
    </div>
    """, unsafe_allow_html=True)


home_tab, graph_tab, recommender_tab1, recommender_tab2, Hakkımızda = st.tabs(["Ana Sayfa", "Ürünler", "Alternatif Ürün Öneri Sistemi", "Cilt Problemine Göre Öneri Sistemi", "Hakkımızda"])



# Ana Sayfa
with home_tab:
    col1, col2, col3 = st.columns([1, 2, 1])
    col1.markdown("""
        <h3 style='font-family: Arial; font-size: 18px; color: #8e44ad;'>
            NEDİR?
        </h3>
        """, unsafe_allow_html=True)
    col1.markdown("""
        <p style='font-family: Arial; font-size: 16px;'>
            CosmetriX, kullanıcıların kozmetik ürün arayışlarını kolaylaştırmak amacıyla geliştirilmiş bir uygulamadır. Kozmetik ürünleri seçerken saatlerce araştırma yapmanıza gerek kalmadan, ihtiyaçlarınıza en uygun ürünleri hızlıca bulmanıza yardımcı olur. İster bütçenize uygun ister cilt tipinize özel ürünler arıyor olun, CosmetriX sizin için en iyi seçenekleri sunar. CosmetriX ile kozmetik alışverişi artık çok daha keyifli ve zahmetsiz!
        </p>
        """, unsafe_allow_html=True)
    col1.image('https://lh3.googleusercontent.com/d/1MAP0EjspPTakQdeedQBeGMDSCimRWLcZ=s400?authuser=0')

    col2.markdown("""
        <h3 style='font-family: Arial; font-size: 18px; color: #8e44ad;'>
            NASIL ÇALIŞIR?
        </h3>
        """, unsafe_allow_html=True)
    col2.markdown("")
    col2.video("https://youtu.be/aKD0jJ_5fWU?si=NDhu-knvsUADFfK_")

    col3.markdown("""
        <h3 style='font-family: Arial; font-size: 18px; color: #8e44ad;'>
            NE İŞE YARAR?
        </h3>
        """, unsafe_allow_html=True)
    col3.markdown("""
        <style>
            ul.custom-list {
                list-style-type: none;
                padding-left: 0;
            }
            ul.custom-list li::before {
                content: "✓"; 
                color: #8e44ad; 
                font-size: 20px; 
                margin-right: 10px;
            }
        </style>
        <p style='font-family: Arial; font-size: 16px;'>
            CosmetriX, kozmetik alışverişlerinizi daha verimli ve keyifli hale getiren bir uygulamadır. İşte CosmetriX'in sunduğu bazı avantajlar:
            <ul class='custom-list' style='font-family: Arial; font-size: 16px;'>
                <li><strong><span style='font-size: 18px;'>Zaman Tasarrufu</span></strong></li>
                <li><strong><span style='font-size: 18px;'>Bütçe Dostu Seçenekler</span></strong></li>
                <li><strong><span style='font-size: 18px;'>Cilt Tipinize Uygun Ürünler</span></strong></li>
                <li><strong><span style='font-size: 18px;'>Kullanıcı Dostu</span></strong></li>
            </ul>
        </p>
        """, unsafe_allow_html=True)
    col3.image('https://lh3.googleusercontent.com/d/1StdekWq6sdEVOUdtEnQ_cIAL1ZbUWAVz=s420?authuser=1')

#ürünler
with graph_tab:
    colg1, colg2 = st.columns(2)

    colg1.markdown("""
            <div style='text-align: center;'>
                <h3 style='font-family: Arial; font-size: 18px; color: #8e44ad;'>
                    ÇOK SATILAN ÜRÜNLER
                </h3>
            </div>
            """, unsafe_allow_html=True)

    fig = px.bar(
        data_frame=veriler.sort_values(by="loves_count", ascending=False).head(10),
        x="reviews",
        y="product_name",
        labels={"reviews": "Yorumlar", "product_name": "Ürün Adı", "rating": "Değerlendirme",
                "price_usd": "Fiyat (USD)", "loves_count": "Beğeni Sayısı", "brand_name": "Marka Adı"},
        orientation="h",
        hover_data=["rating", "price_usd", "loves_count"],
        color="brand_name"
    )
    # Başlık renklerini özelleştirme
    fig.update_layout(
        yaxis_title_font=dict(color='purple'),  # Ürün Adı başlığı
        xaxis_title_font=dict(color='purple')  # Yorumlar başlığı
    )

    colg1.plotly_chart(fig)

    # Marka seçme kutusunun rengini pembe yapmak için stil ekleyin
    st.markdown("""
            <style>
            .stSelectbox > div > div {
                color: purple;
            }
            </style>
            """, unsafe_allow_html=True)

    markalar = veriler["brand_name"].unique().tolist()
    secilen_markalar = colg2.selectbox(label="Marka seçiniz", options=markalar)
    colg2.markdown(f"Seçilen marka: **{secilen_markalar}**")
    filtrelenmis_veriler = veriler[veriler["brand_name"] == secilen_markalar]

    # DataFrame'i HTML tablosuna dönüştürme
    html_table = filtrelenmis_veriler.rename(columns={
        "product_name": "Ürün Adı",
        "brand_name": "Marka Adı",
        "price_usd": "Fiyat (USD)"
    })[["Ürün Adı", "Marka Adı", "Fiyat (USD)"]].to_html(classes='my-table')

    # CSS stilini tanımlama
    css = '''
        <style>
        .my-table {
          border-collapse: collapse;
          width: 100%;
        }

        .my-table th, .my-table td {
          text-align: left;
          padding: 2px;
        }

        .my-table tr:nth-child(even){background-color: #f2f2f2;}
        .my-table tr:nth-child(odd){background-color: #d8bfd8;}
        </style>
        '''

    # HTML tablosunu ve CSS stilini Streamlit'e yazdırma
    colg2.markdown(css, unsafe_allow_html=True)
    colg2.markdown(html_table, unsafe_allow_html=True)



# Ürün Öneri Sistemi Sekmesi
with recommender_tab1:
    st.markdown("""
    <h3 style='font-family: Arial; font-size: 18px; color: #8e44ad;'>
    Alternatif Ürün Öneri Sistemi
    </h3>
    """, unsafe_allow_html=True)

    urun_adi = st.text_input("Ürün Adını Girin")

    if st.button("Önerileri Getir"):
        veriler = preprocess_data(veriler)
        tfidf_matrix = vectorize_data(veriler)
        result = get_recommendations(urun_adi, veriler, tfidf_matrix, n=10)

        if result is not None:
            # DataFrame'i HTML tablosuna dönüştürme
            html_table = result.to_html(classes='my-table')

            # CSS stilini tanımlama
            css = '''
            <style>
            .my-table {
                border-collapse: collapse;
                width: 80%; /* Tablo genişliğini yüzde 80 olarak ayarlama */
                margin-left: auto; 
                margin-right: auto;
            }
            .my-table th, .my-table td {
                text-align: left;
                padding: 2px;
               line-height: 0.9; /* Satır yüksekliğini ayarlama */
            }
            .my-table tr:nth-child(even) { background-color: #f2f2f2; }
            .my-table tr:nth-child(odd) { background-color: #d8bfd8; }
            .image-container {
                text-align: right; /* Sağa hizalama */
            }
            </style>
            '''

            # İki sütunu oluşturma
            col1, col2 = st.columns(2)

            with col1:
                # HTML tablosunu ve CSS stilini Streamlit'e yazdırma
                st.markdown(css, unsafe_allow_html=True)
                st.markdown(html_table, unsafe_allow_html=True)

            with col2:
                # Fotoğrafı ekleme
                st.image('https://lh3.googleusercontent.com/d/1DWnLqzPI2s1Dkgg_yZxoi1U5OBQRnnDK=s480?authuser=0', width=600)


# Yeni Tavsiye Sistemi: Cilt Problemine Göre Öneri Sistemi
def calculate_new_score(row, min_reviews, max_reviews, min_loves, max_loves):
    # Min-Max Normalizasyonu
    normalized_reviews = (row['reviews'] - min_reviews) / (max_reviews - min_reviews)
    normalized_loves = 1 + 4 * (row['loves_count'] - min_loves) / (max_loves - min_loves)

    # Yeni Skoru Hesaplama (rating kolonuna daha fazla ağırlık verilerek)
    raw_score = row['rating'] * 2 * (1 + normalized_reviews) * normalized_loves

    return raw_score

# Skoru 1 ile 5 arasında ölçeklendirme fonksiyonu
def scale_score(raw_score, min_score, max_score):
    return 1 + 4 * (raw_score - min_score) / (max_score - min_score)

# Yeni skoru hesaplayıp veri setine ekleyen fonksiyon
def normalize_and_calculate_scores(df):
    min_reviews = df['reviews'].min()
    max_reviews = df['reviews'].max()
    min_loves = df['loves_count'].min()
    max_loves = df['loves_count'].max()

    df['raw_score'] = df.apply(calculate_new_score, axis=1, args=(min_reviews, max_reviews, min_loves, max_loves))

    min_score = df['raw_score'].min()
    max_score = df['raw_score'].max()

    df['new_score'] = df['raw_score'].apply(scale_score, args=(min_score, max_score))

    return df

# Cilt problemi girdiğinde benzer ürünleri önerme fonksiyonu
def get_recommendations_by_problem(problem, tfidf_vectorizer, tfidf_matrix, skincaredf, cosine_sim=None, top_n=15):
    # Kullanıcının girdiği cilt problemine göre TF-IDF vektörünü hesapla
    problem_tfidf = tfidf_vectorizer.transform([problem])

    # Problemle tüm ürünler arasındaki benzerlik puanlarını al
    sim_scores = cosine_similarity(problem_tfidf, tfidf_matrix).flatten()

    # Benzerlik puanlarına göre ürünleri sırala
    sim_scores_indices = sim_scores.argsort()[::-1][:top_n]

    # En benzer ürünleri al
    recommended_products = skincaredf.iloc[sim_scores_indices].copy()

    # Yeni skoru hesaplama ve veri setine ekleme
    recommended_products = normalize_and_calculate_scores(recommended_products)

    # Yeni skora göre sıralama yap
    recommended_products = recommended_products.sort_values(by='new_score', ascending=False)

    return recommended_products[['product_name_x', 'price_usd', 'rating', 'reviews', 'loves_count', 'problems']]

with recommender_tab2:
    st.header("Cilt Problemine Göre Ürün Öneri Sistemi")

    problem = st.text_input("Cilt probleminizi girin:")

    if st.button("Cilt Problemine Göre Öneri Al"):
        if problem:
            skincaredf = pd.read_csv('pythonProject3/skincaredf.csv')

            # Eksik değerleri dolduralım
            skincaredf[['problem2', 'problem3']] = skincaredf[['problem2', 'problem3']].fillna('')

            # Cilt problemlerini birleştirerek yeni bir sütun oluşturma
            skincaredf['problems'] = skincaredf[['problem1', 'problem2', 'problem3']].agg(','.join, axis=1)

            # TF-IDF hesaplaması için metin alanını oluşturma
            skincaredf['combined_problems'] = skincaredf[['problem1', 'problem2', 'problem3']].fillna('').agg(' '.join,
                                                                                                              axis=1)

            # TF-IDF hesaplaması
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(skincaredf['combined_problems'])

            # Cosine similarity hesaplaması
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


            # Yeni puan hesaplama fonksiyonu
            def calculate_new_score(row, min_reviews, max_reviews, min_loves, max_loves):
                # Min-Max Normalizasyonu
                normalized_reviews = (row['reviews'] - min_reviews) / (max_reviews - min_reviews)
                normalized_loves = 1 + 4 * (row['loves_count'] - min_loves) / (max_loves - min_loves)

                # Yeni Skoru Hesaplama (rating kolonuna daha fazla ağırlık verilerek)
                raw_score = row['rating'] * 2 * (1 + normalized_reviews) * normalized_loves

                return raw_score


            # Skoru 1 ile 5 arasında ölçeklendirme fonksiyonu
            def scale_score(raw_score, min_score, max_score):
                return 1 + 4 * (raw_score - min_score) / (max_score - min_score)


            # Yeni skoru hesaplayıp veri setine ekleyen fonksiyon
            def normalize_and_calculate_scores(df):
                min_reviews = df['reviews'].min()
                max_reviews = df['reviews'].max()
                min_loves = df['loves_count'].min()
                max_loves = df['loves_count'].max()

                df['raw_score'] = df.apply(calculate_new_score, axis=1,
                                           args=(min_reviews, max_reviews, min_loves, max_loves))

                min_score = df['raw_score'].min()
                max_score = df['raw_score'].max()

                df['new_score'] = df['raw_score'].apply(scale_score, args=(min_score, max_score))

                return df


            # Cilt problemi girdiğinde benzer ürünleri önerme fonksiyonu
            def get_recommendations_by_problem(problem, tfidf_vectorizer, tfidf_matrix, skincaredf, cosine_sim=None,
                                               top_n=15):
                # Kullanıcının girdiği cilt problemine göre TF-IDF vektörünü hesapla
                problem_tfidf = tfidf_vectorizer.transform([problem])

                # Problemle tüm ürünler arasındaki benzerlik puanlarını al
                sim_scores = cosine_similarity(problem_tfidf, tfidf_matrix).flatten()

                # Benzerlik puanlarına göre ürünleri sırala
                sim_scores_indices = sim_scores.argsort()[::-1][:top_n]

                # En benzer ürünleri al
                recommended_products = skincaredf.iloc[sim_scores_indices].copy()

                # Yeni skoru hesaplama ve veri setine ekleme
                recommended_products = normalize_and_calculate_scores(recommended_products)

                # Yeni skora göre sıralama yap
                recommended_products = recommended_products.sort_values(by='new_score', ascending=False)

                return recommended_products[
                    ['product_name_x', 'price_usd', 'rating', 'reviews', 'loves_count', 'problems']]


            # Örnek olarak bir cilt problemi vererek benzer ürünleri bulma
            problem = 'Akne'
            recommended_products = get_recommendations_by_problem(problem, tfidf_vectorizer, tfidf_matrix, skincaredf)

            if recommended_products is not None:
                st.write("Önerilen Ürünler:")
                html_code = """
                        <style>
                        .my-table {
                            width: 100%;
                            font-size: 14px;
                        }
                        .my-table tr:nth-child(even) { background-color: #f2f2f2; }
                        .my-table tr:nth-child(odd) { background-color: #d8bfd8; }
                        </style>
                        """
                st.markdown(html_code, unsafe_allow_html=True)

                col1, col2 = st.columns([2, 1])

                with col1:
                    # HTML tabloyu ilk sütunda göster
                    html_table = recommended_products.to_html(classes='my-table', index=False)
                    st.markdown(html_table, unsafe_allow_html=True)

                with col2:
                    # Resmi ikinci sütunda göster
                    st.image("https://lh3.googleusercontent.com/d/1tM21L2bG4vVbi8PnZt4XS7sppKY7ygHC=s520?authuser=1",
                             use_column_width=True)  # Dosya yolunu güncelleyin
            else:
                st.write("Lütfen bir cilt problemi girin.")


# Hakkımızda Sekmesi

with Hakkımızda:
    col1h, col2h = st.columns([1, 1])

    col1h.markdown("""
        <div style='display: flex; justify-content: flex-start; align-items: center; height: 100%;'>
            <img src='https://lh3.googleusercontent.com/d/15RkH5PGcoS2WsA1t8w_WitTFHSrwYHNa=s1080?authuser=0' style='width: 80%; max-width: 500px;'>
        </div>
        """, unsafe_allow_html=True)

    col2h.markdown("""
        <div style='text-align: left; padding-left: 20px;'>
            <h3 style='font-family: Arial; font-size: 18px; color: #8e44ad;'>
                KURUCU ÜYELER
            </h3>
            <p style='font-family: Arial; font-size: 20px;'>
                Berkant: <a href='https://www.linkedin.com/in/mehmetberkantyildiz/'>LinkedIn</a><br>
                Mısra: <a href='https://www.linkedin.com/in/m%C4%B1sray%C4%B1ld%C4%B1r%C4%B1m/'>LinkedIn</a><br>
                Çağıl: <a href='https://www.linkedin.com/in/cagilezgiaydemir/'>LinkedIn</a><br>
                Serhat: <a href='https://www.linkedin.com/in/serhatyurdakul/'>LinkedIn</a><br>
                Sedat: <a href='https://www.linkedin.com/in/sedat-oruc/'>LinkedIn</a><br>
            </p>
        <div style='text-align: left; padding-left: 20px;'>
            <h3 style='font-family: Arial; font-size: 18px; color: #8e44ad;'>
                MANKENLER 
            </h3>
            <p style='font-family: Arial; font-size: 20px;'>
                Vahide Keskine <br>
                Vahit Keskin  <br>
            </p>        
        </div>
        """, unsafe_allow_html=True)