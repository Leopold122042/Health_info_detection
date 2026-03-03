import json
import numpy as np
import jieba
import jieba.analyse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

def gen_tfidf_weights(json_path, output_dir="cache"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    N = len(data)
    num_evd = 5
    tfidf_weights = np.zeros((N, num_evd, 1), dtype=np.float32)

    # 1. 预处理：提取所有文本用于构建语料库
    all_texts = []
    for item in data:
        all_texts.append(" ".join(jieba.cut(item['claim'])))
        for j in range(num_evd):
            content = item['evidence'].get(str(j), {}).get('content', "").strip()
            if content:
                all_texts.append(" ".join(jieba.cut(content)))

    # 2. 训练全语料 TF-IDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_

    print("开始计算基于主体的 TF-IDF 显著性权重...")

    # 3. 计算每个 Evidence 相对于 Claim 的重要性
    for i, item in enumerate(data):
        # 提取 Claim 的关键词及其权重
        claim_keywords = jieba.analyse.extract_tags(item['claim'], topK=10, withWeight=True)
        keyword_map = {word: weight for word, weight in claim_keywords}
        
        for j in range(num_evd):
            content = item['evidence'].get(str(j), {}).get('content', "").strip()
            if not content:
                tfidf_weights[i, j] = 0.0
                continue
            
            evd_words = list(jieba.cut(content))
            score = 0.0
            # 如果证据中出现了声明的关键词，累加该词的 TF-IDF 贡献
            for word in evd_words:
                if word in keyword_map:
                    # 融合关键词在全语料的 IDF 和 在 Claim 中的重要度
                    word_idf = idf[vocab[word]] if word in vocab else 1.0
                    score += keyword_map[word] * word_idf
            
            tfidf_weights[i, j] = score

    # 4. 归一化处理（按样本归一化，突出最相关的证据）
    for i in range(N):
        row_sum = np.sum(tfidf_weights[i])
        if row_sum > 0:
            tfidf_weights[i] = tfidf_weights[i] / row_sum

    # 保存
    Path(output_dir).mkdir(exist_ok=True)
    np.save(f"{output_dir}/tfidf_weights.npy", tfidf_weights)
    print(f"TF-IDF 权重提取完成，已保存至 {output_dir}")

if __name__ == "__main__":
    gen_tfidf_weights("data/health_info_retrieved.json")