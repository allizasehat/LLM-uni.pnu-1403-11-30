import streamlit as st
import fitz  # PyMuPDF برای پردازش PDF
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

# تنظیم صفحه
st.set_page_config(page_title="PDF Chatbot", layout="wide")


# تابع استخراج متن از PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


# تابع پردازش و ذخیره داده در ChromaDB
def process_and_store_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    client = chromadb.PersistentClient(path="./chroma_db")  # ذخیره پایدار
    collection = client.get_or_create_collection(name="pdf_data")

    # نوار پیشرفت
    progress_bar = st.progress(0)

    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"chunk_{i}"])

        # محاسبه درصد پیشرفت
        progress = (i + 1) / len(chunks) * 100
        progress_bar.progress(int(progress))

    return collection


# تابع جستجو و بازیابی متن
def search_db(query):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="pdf_data")

    results = collection.query(query_texts=[query], n_results=3)
    return [doc for doc in results["documents"][0]]


# رابط کاربری Streamlit
st.title("📄 PDF Chatbot with Ollama")

# لیست برای ذخیره تاریخچه چت
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

uploaded_file = st.file_uploader("یک فایل PDF آپلود کنید:", type=["pdf"])

if uploaded_file is not None:
    st.success("فایل آپلود شد! در حال پردازش...")
    text = extract_text_from_pdf(uploaded_file)
    process_and_store_text(text)
    st.success("PDF پردازش و ذخیره شد. حالا می‌توانید سوال بپرسید.")

    query = st.text_input("سوال خود را وارد کنید:")
    if query:
        # ذخیره پرسش کاربر در تاریخچه چت
        st.session_state['messages'].append({"role": "user", "content": query})

        # جستجو در دیتابیس و گرفتن متن مرتبط
        retrieved_texts = search_db(query)
        context = " ".join(retrieved_texts)

        # نمایش درخواست‌ها برای بررسی
        st.write("### پیام‌های ارسال‌شده به مدل:")
        st.write(st.session_state['messages'])

        # استفاده از مدل محلی phi3:mini با شناسه مناسب
        model_id = "phi3:mini"  # اگر مدل محلی باشد، نیازی به مسیر نیست.

        try:
            # ارسال تاریخچه چت به مدل
            response = ollama.chat(model=model_id, messages=st.session_state['messages'] + [
                {"role": "system", "content": "پاسخ خود را بر اساس متن زیر بده:\n" + context}
            ])

            # نمایش پاسخ مدل
            st.write("### پاسخ مدل:")
            st.write(response["message"]["content"])

            # اضافه کردن پاسخ مدل به تاریخچه چت
            st.session_state['messages'].append({"role": "assistant", "content": response["message"]["content"]})
        except Exception as e:
            st.error(f"خطا در ارتباط با مدل: {e}")

        # نمایش تاریخچه چت
        st.write("### تاریخچه چت:")
        for message in st.session_state['messages']:
            if message['role'] == 'user':
                st.markdown(f"**شما:** {message['content']}")
            else:
                st.markdown(f"**مدل:** {message['content']}")
