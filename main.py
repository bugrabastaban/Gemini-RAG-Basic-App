import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# Ortam değişkenlerini (.env dosyasını) yükle (API anahtarları için)
load_dotenv()

# --- 1. Model ve Veri Yükleme (Load) ---

# Gemini Chat Modelini Başlatma (Hızlı ve maliyet etkin olan 'flash' modeli)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Belirtilen URL'den doküman yükleyiciyi tanımla.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs= dict(
        # Yalnızca belirli CSS sınıflarına sahip içeriği (yazı, başlık) çekmek için filtreleme
        parse_only = bs4.SoupStrainer(
            class_ = ("post-content","post-title","post-header")
        )
    )
)

# Dokümanları yükle ve 'docs' değişkenine ata.
docs = loader.load()

# --- 2. Bölme (Split) ve Vektörleştirme (Embed) ---

# Dokümanları parçalayıcıyı tanımla (RAG için küçük parçalar oluşturur)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Her parçanın maksimum boyutu
    chunk_overlap=200 # Parçalar arasındaki çakışma boyutu (bağlamı korumak için)
)
splits = text_splitter.split_documents(docs) # Dokümanları parçalara ayır

# Gemini Embedding Modelini kullanarak parçaları vektörleştir ve ChromaDB'ye kaydet
vectorstore = Chroma.from_documents(
    documents=splits,
    # Embedding modeli: Google'ın önerdiği embedding modelinin tam yolu
    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)

# --- 3. Geri Çağırma (Retrieve) ve Zincirleme (Chain) ---

# Vektör depoyu bir 'retriever' (geri çağırıcı) olarak tanımla
retriever = vectorstore.as_retriever()

# LangChain Hub'dan standart RAG istem şablonunu çek (soru ve bağlamı birleştirir)
prompt = hub.pull("rlm/rag-prompt")

# Geri çağrılan dokümanları string formatına dönüştüren yardımcı fonksiyon
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Zincirini Tanımlama (LCEL - LangChain Expression Language)
rag_chain = (
    # 1. Girdi: "context" (geri çağrılan dokümanlar) ve "question" (kullanıcı sorusu)
    {"context" : retriever | format_docs, "question" : RunnablePassthrough() }
    # 2. Prompta Girdi: Bağlam ve soru ile istem şablonunu doldur
    | prompt
    # 3. LLM'e Gönder: İstem, Gemini'ye gönderilir
    | llm
    # 4. Çıktıyı Çözümle: Gelen yanıtı düz metin (string) olarak al
    | StrOutputParser()
)

# --- 4. Çalıştırma ---

if __name__ == '__main__':
    # RAG zincirini "task decomposition" sorusuyla çalıştır ve çıktıyı stream et (parça parça yazdır)
    print("--- RAG Sonucu ---")
    for chunk in rag_chain.stream("what is task decomposition ?"):
        print(chunk,end="",flush=True)
    print("\n-------------------")