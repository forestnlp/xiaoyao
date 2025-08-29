import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# --- 1. 准备示例文档 ---
# 我们创建了一个包含5个文档的列表。
# 文档 0, 1, 4 是关于金融和投资的。
# 文档 2, 3 是关于人工智能和技术的。
documents = [
    "Stock market analysis shows a bullish trend for tech stocks.",
    "Investing in bonds is a safe bet for long-term financial growth.",
    "Deep learning and neural networks are the core of modern artificial intelligence.",
    "Artificial intelligence and machine learning are transforming industries.",
    "The federal reserve announced a change in interest rates, affecting the stock market."
]

# --- 2. 文本预处理 ---
stop_words = set(stopwords.words('english'))

processed_docs = []
for doc in documents:
    # 分词
    tokens = word_tokenize(doc.lower())
    # 去除停用词和短于3个字符的词
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 2]
    processed_docs.append(filtered_tokens)

print("--- 预处理后的文档 ---")
for i, doc in enumerate(processed_docs):
    print(f"文档 {i}: {doc}")
print("\n")

# --- 3. 创建词典和语料库 (Bag-of-Words) ---
# 创建一个词语到ID的映射
dictionary = Dictionary(processed_docs)

# 将每个文档转换为词袋模型格式（(词ID, 词频)）
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# --- 4. 训练 LDA 模型 ---
# num_topics=2: 我们告诉模型我们想发现2个主题。
# id2word=dictionary: 将ID映射回词语，便于解读。
# passes=15: 训练的轮次。
print("--- 开始训练 LDA 模型 ---")
lda_model = LdaModel(
    corpus=corpus, 
    id2word=dictionary, 
    num_topics=2, 
    random_state=100, 
    passes=15
)
print("--- 模型训练完成 ---\n")

# --- 5. 查看发现的主题 ---
print("--- LDA 模型发现的主题 ---")
topics = lda_model.print_topics(num_words=5) # 每个主题显示5个最重要的词
for topic in topics:
    print(topic)

# --- 6. 查看单个文档的主题分布 ---
print("\n--- 单个文档的主题分布 ---")
for i, doc_bow in enumerate(corpus):
    print(f"文档 {i}: {lda_model.get_document_topics(doc_bow)}")
