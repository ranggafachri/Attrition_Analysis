
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve, roc_auc_score,
                           classification_report, silhouette_score)

st.set_page_config(
    page_title="Dashboard Analisis Attrition Karyawan",
    page_icon="⚒️",
    layout="wide"
)

st.title("Dashboard Analisis Attrition Karyawan")
st.write("Aplikasi untuk menganalisis dan memprediksi tingkat attrition karyawan menggunakan Machine Learning")

st.sidebar.title("Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Analisis:",
    ["Data Overview", "Model Evaluation", "Prediction Tool"]
)

@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv("Dataset_Karyawan.csv")
    except:
        st.error("File Dataset_Karyawan.csv tidak ditemukan. Menggunakan data simulasi.")
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'Age': np.random.randint(22, 65, n_samples),
            'DistanceFromHome': np.random.randint(1, 30, n_samples),
            'Attrition': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'JobSatisfaction': np.random.randint(1, 5, n_samples),
            'MonthlyIncome': np.random.randint(2000, 20000, n_samples),
            'MonthlyRate': np.random.randint(2000, 25000, n_samples),
            'OverTime': np.random.choice(['Yes', 'No'], n_samples),
            'YearsAtCompany': np.random.randint(0, 40, n_samples),
            'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], n_samples),
            'JobRole': np.random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manager'], n_samples),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples)
        })

    selected_col = ['Age', 'DistanceFromHome', 'Attrition', 'Gender', 'JobSatisfaction',
                   'MonthlyIncome', 'MonthlyRate', 'OverTime', 
                   'YearsAtCompany', 'BusinessTravel', 'JobRole', 'MaritalStatus']

    df = df[selected_col]

    df_num = df.select_dtypes(include=['number'])
    df_cat = df.select_dtypes(exclude=['number'])

    Q1 = df_num.quantile(0.25)
    Q3 = df_num.quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    outlier = ((df_num >= batas_bawah) & (df_num <= batas_atas)).all(axis=1)
    clean_df = pd.concat([df_num[outlier], df_cat[outlier]], axis=1)

    return df, clean_df

@st.cache_data
def train_model(clean_df):
    df_num = clean_df.copy()
    labele = LabelEncoder()

    for col in clean_df.columns:
        if clean_df[col].dtype == object:
            df_num[col] = labele.fit_transform(clean_df[col])

    y = df_num['Attrition']
    X = df_num.drop(['Attrition'], axis=1)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    y_pred = nb_model.predict(X_test)
    y_prob = nb_model.predict_proba(X_test)[:, 1]

    return nb_model, X_test, y_test, y_pred, y_prob, X.columns, labele

@st.cache_data
def perform_clustering(clean_df):
    df_num = clean_df.copy()
    labele = LabelEncoder()

    for col in clean_df.columns:
        if clean_df[col].dtype == object:
            df_num[col] = labele.fit_transform(clean_df[col])

    num_col = ["MonthlyIncome", "YearsAtCompany", "MonthlyRate"]
    df_selected = df_num[num_col]

    scaler = StandardScaler()
    hasil_scale = scaler.fit_transform(df_selected)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(hasil_scale)

    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(hasil_scale)

    silhouette_avg = silhouette_score(hasil_scale, labels)

    return df_pca, labels, silhouette_avg, kmeans, scaler, pca

df, clean_df = load_and_prepare_data()

if menu == "Data Overview":
    st.header("Data Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Jumlah Data:** {df.shape[0]} baris, {df.shape[1]} kolom")
        st.write(f"**Data Setelah Cleaning:** {clean_df.shape[0]} baris")

        st.subheader("Distribusi Attrition")
        attrition_counts = clean_df['Attrition'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', 
                colors=['skyblue', 'lightcoral'])
        ax1.set_title('Distribusi Attrition')
        st.pyplot(fig1)

    with col2:
        st.subheader("Statistik Deskriptif")
        st.dataframe(clean_df.describe())

        st.subheader("Sample Data")
        st.dataframe(clean_df.head())

    st.subheader("Analisis Korelasi")
    df_corr = clean_df.copy()
    labele = LabelEncoder()

    for col in df_corr.columns:
        if df_corr[col].dtype == object:
            df_corr[col] = labele.fit_transform(df_corr[col])

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_corr.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
    ax2.set_title('Correlation Matrix')
    st.pyplot(fig2)

elif menu == "Model Evaluation":
    st.header("Evaluasi Model Machine Learning")

    model, X_test, y_test, y_pred, y_prob, feature_names, label_encoder = train_model(clean_df)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", f"{acc:.3f}")
    with col2:
        st.metric("Precision", f"{prec:.3f}")
    with col3:
        st.metric("Recall", f"{rec:.3f}")
    with col4:
        st.metric("F1-Score", f"{f1:.3f}")
    with col5:
        st.metric("ROC AUC", f"{roc_auc:.3f}")

    plot_option = st.selectbox("Pilih visualisasi:", ["ROC Curve", "Confusion Matrix", "K-Means Clustering"])

    if plot_option == "ROC Curve":
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', color='red', lw=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Naive Bayes')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    elif plot_option == "Confusion Matrix":
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    elif plot_option == "K-Means Clustering":
        st.header("Analisis Clustering")

        df_pca, labels, silhouette_avg, kmeans, scaler, pca = perform_clustering(clean_df)

        st.write(f"**Silhouette Score:** {silhouette_avg:.4f}")

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', s=50)

        centroid_pca = pca.transform(kmeans.cluster_centers_)
        ax.scatter(centroid_pca[:, 0], centroid_pca[:, 1], c='red', s=200, marker='*', 
                label='Centroids', edgecolors='black')

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title(f'K-Means Clustering (Silhouette Score: {silhouette_avg:.3f})')
        ax.legend()
        ax.grid(True)
        plt.colorbar(scatter)
        st.pyplot(fig)

elif menu == "Prediction Tool":
    st.header("Tool Prediksi Attrition")

    model, _, _, _, _, feature_names, label_encoder = train_model(clean_df)

    st.write("Masukkan data karyawan untuk memprediksi kemungkinan attrition:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Usia:", 18, 65, 30)
            distance = st.slider("Jarak dari Rumah (km):", 1, 30, 10)
            gender = st.selectbox("Gender:", ["Male", "Female"])
            job_satisfaction = st.slider("Job Satisfaction (1-4):", 1, 4, 3)
            monthly_income = st.number_input("Monthly Income:", 1000, 50000, 5000)
            monthly_rate = st.number_input("Monthly Rate:", 1000, 30000, 15000)

        with col2:
            overtime = st.selectbox("Overtime:", ["Yes", "No"])
            years_company = st.slider("Years at Company:", 0, 40, 5)
            business_travel = st.selectbox("Business Travel:", 
                                         ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            job_role = st.selectbox("Job Role:", 
                                   ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager"])
            marital_status = st.selectbox("Marital Status:", ["Single", "Married", "Divorced"])

        submit = st.form_submit_button("Prediksi Attrition")

    if submit:
        input_data = pd.DataFrame({
            'Age': [age],
            'DistanceFromHome': [distance],
            'Gender': [gender],
            'JobSatisfaction': [job_satisfaction],
            'MonthlyIncome': [monthly_income],
            'MonthlyRate': [monthly_rate],
            'OverTime': [overtime],
            'YearsAtCompany': [years_company],
            'BusinessTravel': [business_travel],
            'JobRole': [job_role],
            'MaritalStatus': [marital_status]
        })

        input_encoded = input_data.copy()
        labele = LabelEncoder()

        temp_df = clean_df.copy()
        for col in temp_df.columns:
            if temp_df[col].dtype == object and col != 'Attrition':
                if col in input_encoded.columns:
                    le_temp = LabelEncoder()
                    le_temp.fit(temp_df[col])
                    try:
                        input_encoded[col] = le_temp.transform(input_encoded[col])
                    except:
                        input_encoded[col] = 0  

        input_dummies = pd.get_dummies(input_encoded, drop_first=True)

        for col in feature_names:
            if col not in input_dummies.columns:
                input_dummies[col] = 0

        input_dummies = input_dummies[feature_names]

        prediction = model.predict(input_dummies)[0]
        probability = model.predict_proba(input_dummies)[0]

        st.subheader("Hasil Prediksi")

        if prediction == 1:
            st.error(f"**Karyawan Berpotensi Mengalami Attrition**")
            st.write(f"Probabilitas Attrition: **{probability[1]:.2%}**")
        else:
            st.success(f"**Karyawan Tidak Berpotensi Mengalami Attrition**")
            st.write(f"Probabilitas Tetap Bekerja: **{probability[0]:.2%}**")

        risk_level = "Tinggi" if probability[1] > 0.7 else "Sedang" if probability[1] > 0.4 else "Rendah"
        st.write(f"Tingkat Risiko: **{risk_level}**")
