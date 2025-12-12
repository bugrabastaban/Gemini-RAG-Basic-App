# 🤖 Gemini ve LangChain ile Basit RAG Uygulaması (Öğrenme Projesi)

Bu depo, **Retrieval Augmented Generation (RAG)** mimarisinin temel yapısını öğrenme ve uygulama amacıyla oluşturulmuştur. Google'ın **Gemini** büyük dil modeli ile **LangChain** çatısının entegrasyonunu göstermektedir.

Bu projede, harici bir web sayfasından veri çekilip vektörleştirilerek, modelin sadece kendi bilgisi yerine güncel ve bağlamsal bilgi kullanarak cevap vermesi sağlanmıştır.

## 🎯 Proje Amacı

* **Öğrenme Odaklı:** LangChain Expression Language (LCEL) ve RAG zinciri oluşturma süreçlerini pratik etmek.
* **LangChain Bileşenleri:** `WebBaseLoader`, `RecursiveCharacterTextSplitter`, `Chroma` Vektör Deposu ve `ChatGoogleGenerativeAI` gibi temel bileşenlerin işleyişini anlamak.
* **Çok Aşamalı İş Akışı:** Doküman yükleme, parçalama, vektörleştirme ve sorgulama adımlarını bir zincirde birleştirmeyi göstermek.

## 🛠️ Kullanılan Teknolojiler

* **Büyük Dil Modeli (LLM):** Gemini 2.5 Flash
* **Çatı (Framework):** LangChain
* **Embedding Modeli:** `models/text-embedding-004` (Google Generative AI)
* **Vektör Deposu:** ChromaDB (Yerel olarak in-memory)
* **Veri Kaynağı:** [Lilian Weng'in "LLM Powered Autonomous Agents" Makalesi](https://lilianweng.github.io/posts/2023-06-23-agent/)

## 🚀 Kurulum ve Çalıştırma

### 1. Ön Gereksinimler

* Python 3.10+
* Google Gemini API Anahtarı (`AIzaSy...`)

### 2. Ortam Hazırlığı

Proje dizininde (repo'yu klonladığınız yerde) aşağıdaki paketleri kurun:

```bash
pip install langchain langchain-google-genai langchain-chroma python-dotenv bs4
