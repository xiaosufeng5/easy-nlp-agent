
import streamlit as st
import os
import sys
import tempfile


from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 修正后的 app.py 导入语句 (最终版本) ---

# 1. 导入 LangChain Chains 依赖 (从 langchain-community 尝试导入)
# ⚠️ 注意：RetrievalQA 在最新版本中被移到了 community 包
from langchain_community.chains import RetrievalQA 

# 2. 导入 LangChain Core 依赖
from langchain_core.prompts import PromptTemplate # ⬅️ 将 PromptTemplate 移到 core 包

# 3. 导入 LangChain Community 依赖 (保持不变)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# --- 2. 配置和初始化 (Streamlit Caching) ---

# 部署时，需将 PDF 文件放在 /data 目录下
DATA_PATH = "data/NLP and Text Analysis: Introduction.pdf"
PERSIST_DIR = "./chroma_db_cache"  # 数据库缓存目录

# ⚠️ 部署到 Streamlit Cloud 时，API Key 必须通过 Secrets 传入
API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY

# 初始化模型 (使用 Streamlit 的 Caching 机制确保只运行一次)
@st.cache_resource
def initialize_models():
    # LLM (用于生成答案)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        convert_system_message_to_human=True,
        google_api_key=API_KEY
    )
    # Embedding Model (用于建立和查询向量)
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    return llm, embedding_model

llm, embedding_model = initialize_models()


# --- 3. 核心：建立或加载 Vector Store (数据处理) ---

@st.cache_resource
def setup_vector_store():
    # 尝试加载已存在的数据库
    try:
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
        # 简单测试，确保数据库不是空的
        if db._collection.count() > 0:
            st.success("✅ 已加载持久化知识库。")
            return db
    except:
        pass # 如果加载失败或不存在，则重新建立

    # 如果数据库不存在，则从头建立
    st.info("🔄 知识库不存在或为空，正在从 PDF 文件建立索引 (这只会发生一次)...")
    
    # 1. 加载文件
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    
    # 2. 知识切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. 向量化和存储
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=PERSIST_DIR 
    )
    st.success(f"✅ 知识库建立完成！共索引 {db._collection.count()} 个知识块。")
    return db

db = setup_vector_store()

# --- 4. Prompt 定义和 Q&A 链设置 ---

TEMPLATE = """
## I. Role and Persona (System Role)
You are a **highly constrained, specialized University Teaching Assistant**. Your **SOLE** source of knowledge is the provided 【Reference Material】. You **MUST NOT** use any external knowledge, common sense, or pre-existing training data. Your primary goal is to ensure the student's answer is verifiable within the provided context.

## II. RAG Action Rules (The Logic)
1. **Source Constraint:** Your answers **MUST** be entirely and **EXCLUSIVELY** based on the content found in the 【Reference Material】. If the document does not contain a specific answer, you must proceed to the Refusal Rule immediately.
2. **Refusal Rule:** If the 【Reference Material】 does not cover the exact question, you **MUST** politely decline and state: "I apologize, this specific information is not covered in the course material." Do not guess or infer.
3. **Comparison Rule:** If the student asks about a difference or comparison between two concepts (e.g., two algorithms or methods), you must present the answer using a clear, side-by-side comparison (table or bullet points).

## III. Knowledge Insertion
【Reference Material】:
{context}

## IV. Student Query
{question}
"""

CUSTOM_PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

retriever = db.as_retriever(search_kwargs={"k": 4}) 

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=retriever,
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}, 
    return_source_documents=False 
)


# --- 5. Streamlit 界面逻辑 ---
# --- 5. Streamlit 界面逻辑 (完整交互版本) ---

st.title("📚 NLP AI Agent (RAG)")
st.caption("✅ 知识库已从本地文件加载。")

# 1. 初始化聊天记录 (Chat History)
# 确保聊天记录在会话状态中是持久化的
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是你的 NLP 课程助教，请问有什么关于课程的问题吗？"}
    ]

# 2. 显示所有历史消息
for msg in st.session_state.messages:
    # 使用 st.chat_message 自动美化消息气泡
    st.chat_message(msg["role"]).write(msg["content"])

# 3. 接收用户输入 (Input)
# st.chat_input 会自动创建一个输入框并处理用户输入
if prompt := st.chat_input("输入你的问题..."):
    # 检查 API Key 是否设置
    if not API_KEY:
        st.error("部署错误：请在 Streamlit Secrets 中设置 GOOGLE_API_KEY。")
        # 立即停止执行后续逻辑
        sys.exit() 
    
    # a. 将用户输入添加到历史记录并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # b. 调用 Agent 进行检索和生成 (Output)
    with st.spinner("🧠 助教正在查阅知识库并思考..."):
        # ⚠️ 注意：这里你需要调用之前在 app.py 中定义的 qa_chain.invoke
        # 确保这个函数在当前作用域可用
        try:
            # 调用 qa_chain (我们假设它是可用的)
            result = qa_chain.invoke({"query": prompt})
            response = result['result']
        except Exception as e:
            response = f"❌ 内部错误：Agent 无法处理请求。错误详情: {e}"

    # c. 将 Agent 回答添加到历史记录并显示
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
