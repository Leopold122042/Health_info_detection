import json
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg as pseg
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def load_stopwords(stopwords_path: str):
    p = Path(stopwords_path)
    if not p.exists():
        return set()
    stops = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            stops.add(w)
    return stops


def tokenize(text: str, stopwords: set):
    """jieba 分词 + 停用词/噪声过滤。"""
    toks = []
    for w in jieba.cut(text):
        w = w.strip()
        if not w:
            continue
        if w in stopwords:
            continue
        if len(w) < 2:
            continue
        if _PUNCT_RE.match(w):
            continue
        if w.isdigit():
            continue
        toks.append(w)
    return toks


def extract_noun_terms(text: str, stopwords: set):
    """
    从 claim 中提取“名词类关键词”集合。
    - jieba 词性标注：n/nr/ns/nt/nz... 通常都以 'n' 开头
    - 同时过滤停用词、短 token、纯标点/数字
    """
    nouns = set()
    for w, flag in pseg.cut(text):
        w = w.strip()
        if not w:
            continue
        if w in stopwords:
            continue
        if len(w) < 2:
            continue
        if _PUNCT_RE.match(w) or w.isdigit():
            continue
        if flag and (flag.startswith("n") or flag in {"vn"}):
            nouns.add(w)
    return nouns


def gen_tfidf_weights(
    json_path,
    output_dir="cache",
    stopwords_path="data/stopwords_zh.txt",
    topk_nouns=20,
):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    N = len(data)
    num_evd = 5
    tfidf_weights = np.zeros((N, num_evd, 1), dtype=np.float32)

    stopwords = load_stopwords(stopwords_path)
    if stopwords:
        print(f"Loaded stopwords: {len(stopwords)} from {stopwords_path}")
    else:
        print(f"[Warn] stopwords file not found or empty: {stopwords_path} (continue without stops)")

    # 1. 预处理：提取所有文本用于构建语料库
    all_texts = []
    for item in data:
        all_texts.append(" ".join(tokenize(item['claim'], stopwords)))
        for j in range(num_evd):
            content = item['evidence'].get(str(j), {}).get('content', "").strip()
            if content:
                all_texts.append(" ".join(tokenize(content, stopwords)))

    # 2. 训练全语料 TF-IDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)
    terms = vectorizer.get_feature_names_out()

    print("开始计算（停用词过滤 + claim 名词关键词优先）的 TF-IDF 权重...")

    # 3. 计算每个 Evidence 相对于 Claim 的重要性
    for i, item in enumerate(data):
        claim = item["claim"]
        claim_nouns = extract_noun_terms(claim, stopwords)

        # 使用 TF-IDF 向量从 claim 中挑出 top-k 的“名词”关键词及其 tf-idf 权重
        claim_tokens = " ".join(tokenize(claim, stopwords))
        claim_vec = vectorizer.transform([claim_tokens])
        claim_arr = claim_vec.toarray()[0]
        if claim_arr.sum() == 0:
            keyword_map = {}
        else:
            top_idx = np.argsort(-claim_arr)
            keyword_map = {}
            for idx in top_idx:
                if claim_arr[idx] <= 0:
                    break
                w = terms[idx]
                if w in claim_nouns:
                    keyword_map[w] = float(claim_arr[idx])
                    if len(keyword_map) >= topk_nouns:
                        break

        # 若名词关键词为空，回退到 extract_tags（同样做停用词过滤）
        if not keyword_map:
            claim_keywords = jieba.analyse.extract_tags(claim, topK=10, withWeight=True)
            keyword_map = {
                w: float(weight)
                for w, weight in claim_keywords
                if w and (w not in stopwords) and len(w) >= 2 and (not _PUNCT_RE.match(w)) and (not w.isdigit())
            }
        
        for j in range(num_evd):
            content = item['evidence'].get(str(j), {}).get('content', "").strip()
            if not content:
                tfidf_weights[i, j] = 0.0
                continue
            
            evd_words = set(tokenize(content, stopwords))
            score = 0.0
            # 若证据中出现声明关键词（优先名词），累加该词在 claim 中的 tf-idf 权重
            for word, w_weight in keyword_map.items():
                if word in evd_words:
                    score += w_weight
            
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