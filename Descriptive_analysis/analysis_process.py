import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

import numpy as np


def load_json(path):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def load_stopwords(path):
	if not path or not os.path.exists(path):
		return set()
	with open(path, "r", encoding="utf-8") as f:
		return {line.strip() for line in f if line.strip()}


def ensure_keyword_blocklist(path):
	default_words = [
		"谣言",
		"辟谣",
		"真的",
		"假的",
		"可以",
		"不会",
		"不是",
		"表示",
		"相关",
		"进行",
		"一个",
		"没有",
		"问题",
		"情况",
		"影响",
	]
	if not os.path.exists(path):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, "w", encoding="utf-8") as f:
			f.write("# 关键词禁用列表（每行一个词，支持你自由补充）\n")
			f.write("# 以 # 开头的行为注释\n")
			for w in default_words:
				f.write(f"{w}\n")


def load_keyword_blocklist(path):
	if not path:
		return set()
	ensure_keyword_blocklist(path)
	words = set()
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			t = line.strip()
			if not t or t.startswith("#"):
				continue
			words.add(t)
	return words


def parse_date(date_str):
	if not date_str:
		return None
	try:
		return datetime.strptime(date_str, "%Y-%m-%d")
	except Exception:
		return None


def safe_int(x, default=None):
	try:
		return int(x)
	except Exception:
		return default


def get_evidence_texts(item):
	evd = item.get("evidence", {})
	texts = []
	if isinstance(evd, dict):
		for key in sorted(evd.keys(), key=lambda t: safe_int(t, 999)):
			value = evd.get(key, {})
			if isinstance(value, dict):
				content = (value.get("content") or "").strip()
			else:
				content = str(value).strip()
			texts.append(content)
	return texts


def clean_text(text):
	if not text:
		return ""
	text = str(text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def basic_summary_stats(values):
	if not values:
		return {
			"count": 0,
			"mean": None,
			"median": None,
			"min": None,
			"p25": None,
			"p75": None,
			"max": None,
		}
	arr = np.array(values)
	return {
		"count": int(arr.size),
		"mean": float(arr.mean()),
		"median": float(np.median(arr)),
		"min": int(arr.min()),
		"p25": float(np.percentile(arr, 25)),
		"p75": float(np.percentile(arr, 75)),
		"max": int(arr.max()),
	}


def histogram(values, bins=20):
	if not values:
		return {"bin_edges": [], "counts": []}
	counts, edges = np.histogram(values, bins=bins)
	return {
		"bin_edges": [float(x) for x in edges.tolist()],
		"counts": [int(x) for x in counts.tolist()],
	}


def tokenize(text, stopwords, blocklist):
	text = clean_text(text)
	if not text:
		return []

	tokens = []
	try:
		import jieba

		for token in jieba.cut(text):
			t = token.strip().lower()
			if not t:
				continue
			if t in stopwords:
				continue
			if t in blocklist:
				continue
			if re.fullmatch(r"[\W_]+", t):
				continue
			if len(t) == 1 and re.fullmatch(r"[\u4e00-\u9fff]", t):
				continue
			tokens.append(t)
	except Exception:
		filtered = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text.lower())
		for i in range(max(0, len(filtered) - 1)):
			t = filtered[i : i + 2]
			if t in stopwords:
				continue
			if t in blocklist:
				continue
			tokens.append(t)

	return tokens


def top_keywords_by_period(records, stopwords, blocklist, top_n=30):
	period_counters = defaultdict(Counter)
	period_sizes = Counter()
	for r in records:
		dt = r.get("parsed_date")
		if dt is None:
			continue
		period = f"{dt.year}"
		period_sizes[period] += 1
		for token in tokenize(r.get("claim", ""), stopwords, blocklist):
			period_counters[period][token] += 1

	result = {}
	for period in sorted(period_counters.keys()):
		result[period] = {
			"sample_size": int(period_sizes[period]),
			"top_keywords": [
				{"token": token, "freq": int(freq)}
				for token, freq in period_counters[period].most_common(top_n)
			],
		}
	return result


def distorted_claim_topics(records, stopwords, blocklist, top_n=40):
	c1 = Counter()
	c0 = Counter()

	for r in records:
		label = r.get("label")
		tokens = tokenize(r.get("claim", ""), stopwords, blocklist)
		if label == 1:
			c1.update(tokens)
		elif label == 0:
			c0.update(tokens)

	v = set(c1.keys()) | set(c0.keys())
	n1 = sum(c1.values())
	n0 = sum(c0.values())
	if not v or n1 == 0:
		return {
			"top_by_log_odds": [],
			"meta": {"label1_total_tokens": int(n1), "label0_total_tokens": int(n0)},
		}

	alpha = 1.0
	denom1 = n1 + alpha * len(v)
	denom0 = n0 + alpha * len(v)

	rows = []
	for token in v:
		p1 = (c1[token] + alpha) / denom1
		p0 = (c0[token] + alpha) / denom0
		ratio = p1 / p0 if p0 > 0 else float("inf")
		rows.append(
			{
				"token": token,
				"freq_label1": int(c1[token]),
				"freq_label0": int(c0[token]),
				"ratio_label1_over_label0": float(ratio),
				"log_odds": float(math.log(ratio)) if ratio > 0 else None,
			}
		)

	rows.sort(key=lambda x: (x["log_odds"], x["freq_label1"]), reverse=True)
	return {
		"top_by_log_odds": rows[:top_n],
		"meta": {"label1_total_tokens": int(n1), "label0_total_tokens": int(n0)},
	}


def safe_scipy_stats():
	try:
		from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu

		return chi2_contingency, ks_2samp, mannwhitneyu, None
	except Exception as e:
		return None, None, None, str(e)


def p_to_stars(p_value):
	if p_value is None:
		return "n.s."
	try:
		p = float(p_value)
	except Exception:
		return "n.s."
	if p < 0.001:
		return "***"
	if p < 0.01:
		return "**"
	if p < 0.05:
		return "*"
	return "n.s."


def fmt_num(v, nd=4):
	if v is None:
		return "NA"
	try:
		val = float(v)
		if abs(val) < 1e-3 and val != 0:
			return f"{val:.2e}"
		return f"{val:.{nd}f}"
	except Exception:
		return str(v)


def print_table(title, headers, rows):
	print("\n" + "=" * 96)
	print(title)
	print("=" * 96)
	if not rows:
		print("(empty)")
		return
	str_rows = [[str(c) for c in row] for row in rows]
	widths = [len(h) for h in headers]
	for row in str_rows:
		for i, cell in enumerate(row):
			widths[i] = max(widths[i], len(cell))

	def _line(parts):
		return " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts)))

	print(_line(headers))
	print("-+-".join("-" * w for w in widths))
	for row in str_rows:
		print(_line(row))


def evidence_slot_stats(records):
	overall = Counter(r["evidence_slots"] for r in records)
	by_label = defaultdict(Counter)
	for r in records:
		by_label[r["label"]][r["evidence_slots"]] += 1

	slot_levels = list(range(0, 6))
	contingency = []
	for label in [0, 1]:
		contingency.append([by_label[label].get(s, 0) for s in slot_levels])

	chi2_contingency, _, mannwhitneyu, scipy_error = safe_scipy_stats()
	significance = {
		"chi_square": None,
		"mann_whitney_u": None,
		"mann_whitney_u_label0_less_label1": None,
		"scipy_error": scipy_error,
	}

	if chi2_contingency is not None:
		try:
			chi2, p, dof, exp = chi2_contingency(np.array(contingency))
			significance["chi_square"] = {
				"chi2": float(chi2),
				"p_value": float(p),
				"dof": int(dof),
				"expected": [[float(v) for v in row] for row in exp.tolist()],
			}
		except Exception as e:
			significance["chi_square"] = {"error": str(e)}

	if mannwhitneyu is not None:
		try:
			s0 = [r["evidence_slots"] for r in records if r["label"] == 0]
			s1 = [r["evidence_slots"] for r in records if r["label"] == 1]
			if s0 and s1:
				u_stat, p_val = mannwhitneyu(s0, s1, alternative="two-sided")
				significance["mann_whitney_u"] = {
					"u_stat": float(u_stat),
					"p_value": float(p_val),
					"n_label0": len(s0),
					"n_label1": len(s1),
					"median_label0": float(median(s0)),
					"median_label1": float(median(s1)),
				}
				u_less, p_less = mannwhitneyu(s0, s1, alternative="less")
				significance["mann_whitney_u_label0_less_label1"] = {
					"u_stat": float(u_less),
					"p_value": float(p_less),
					"n_label0": len(s0),
					"n_label1": len(s1),
					"mean_label0": float(np.mean(s0)),
					"mean_label1": float(np.mean(s1)),
					"median_label0": float(median(s0)),
					"median_label1": float(median(s1)),
					"hypothesis": "label0 < label1",
				}
		except Exception as e:
			significance["mann_whitney_u"] = {"error": str(e)}

	return {
		"overall_distribution": {str(k): int(v) for k, v in sorted(overall.items())},
		"by_label_distribution": {
			str(label): {str(k): int(v) for k, v in sorted(cnt.items())}
			for label, cnt in by_label.items()
		},
		"contingency_table": {
			"slot_levels": slot_levels,
			"label_0": contingency[0],
			"label_1": contingency[1],
		},
		"significance": significance,
	}


def tfidf_similarity_stats(records):
	claim_texts = [r.get("claim", "") for r in records]
	evidence_texts = [r.get("evidence_concat", "") for r in records]

	out = {
		"available": False,
		"error": None,
		"overall": None,
		"by_label": {},
		"ks_test_label0_vs_label1": None,
		"mann_whitney_u_label0_greater_label1": None,
	}

	try:
		from sklearn.feature_extraction.text import TfidfVectorizer
		from sklearn.metrics.pairwise import cosine_similarity
	except Exception as e:
		out["error"] = f"scikit-learn not available: {e}"
		return out

	vectorizer = TfidfVectorizer(min_df=2)
	corpus = claim_texts + evidence_texts
	x = vectorizer.fit_transform(corpus)
	n = len(claim_texts)
	claim_vec = x[:n]
	evd_vec = x[n:]
	sims = cosine_similarity(claim_vec, evd_vec).diagonal()

	sim_list = [float(v) for v in sims.tolist()]
	out["available"] = True
	out["overall"] = {
		"summary": basic_summary_stats(sim_list),
		"histogram": histogram(sim_list, bins=20),
		"values": sim_list,
	}

	for label in [0, 1]:
		label_vals = [float(sims[i]) for i, r in enumerate(records) if r["label"] == label]
		out["by_label"][str(label)] = {
			"summary": basic_summary_stats(label_vals),
			"histogram": histogram(label_vals, bins=20),
			"values": label_vals,
		}

	_, ks_2samp, _, scipy_error = safe_scipy_stats()
	chi2_contingency, ks_2samp, mannwhitneyu, scipy_error = safe_scipy_stats()
	if ks_2samp is not None:
		try:
			s0 = out["by_label"].get("0", {}).get("values", [])
			s1 = out["by_label"].get("1", {}).get("values", [])
			if s0 and s1:
				ks = ks_2samp(s0, s1)
				out["ks_test_label0_vs_label1"] = {
					"statistic": float(ks.statistic),
					"p_value": float(ks.pvalue),
				}
			if mannwhitneyu is not None:
				u_greater, p_greater = mannwhitneyu(s0, s1, alternative="greater")
				out["mann_whitney_u_label0_greater_label1"] = {
					"u_stat": float(u_greater),
					"p_value": float(p_greater),
					"mean_label0": float(np.mean(s0)) if s0 else None,
					"mean_label1": float(np.mean(s1)) if s1 else None,
					"median_label0": float(median(s0)) if s0 else None,
					"median_label1": float(median(s1)) if s1 else None,
					"hypothesis": "label0 > label1",
				}
		except Exception as e:
			out["ks_test_label0_vs_label1"] = {"error": str(e)}
	else:
		out["ks_test_label0_vs_label1"] = {"error": scipy_error}

	return out


def evidence_sentiment_stats(records):
	out = {
		"available": False,
		"error": None,
		"overall": None,
		"by_label": {},
		"mann_whitney_u_label0_vs_label1": None,
		"method": "lexicon_ratio",
	}

	positive_words = {
		"有效",
		"安全",
		"可靠",
		"正常",
		"改善",
		"康复",
		"治愈",
		"有益",
		"预防",
		"支持",
		"稳定",
		"增强",
		"健康",
		"合格",
	}
	negative_words = {
		"风险",
		"危险",
		"有害",
		"恶化",
		"传播",
		"感染",
		"死亡",
		"副作用",
		"焦虑",
		"恐慌",
		"失效",
		"异常",
		"中毒",
		"致癌",
	}

	def _simple_sentiment_score(text):
		text = clean_text(text)
		if not text:
			return None
		try:
			import jieba

			tokens = [t.strip().lower() for t in jieba.cut(text) if t.strip()]
		except Exception:
			tokens = re.findall(r"[\u4e00-\u9fffA-Za-z]+", text.lower())

		pos = 0
		neg = 0
		for token in tokens:
			if token in positive_words:
				pos += 1
			if token in negative_words:
				neg += 1

		total = pos + neg
		if total == 0:
			return 0.5
		raw = (pos - neg) / total
		return float((raw + 1.0) / 2.0)

	scores = []
	for r in records:
		text = r.get("evidence_concat", "")
		if not text:
			scores.append(None)
			continue
		s = _simple_sentiment_score(text)
		scores.append(s)

	valid_scores = [s for s in scores if s is not None]
	out["available"] = True
	out["overall"] = {
		"summary": basic_summary_stats(valid_scores),
		"histogram": histogram(valid_scores, bins=20),
		"values": [float(v) for v in valid_scores],
	}

	for label in [0, 1]:
		label_vals = [
			float(scores[i])
			for i, r in enumerate(records)
			if r["label"] == label and scores[i] is not None
		]
		out["by_label"][str(label)] = {
			"summary": basic_summary_stats(label_vals),
			"histogram": histogram(label_vals, bins=20),
			"values": label_vals,
		}

	_, _, mannwhitneyu, scipy_error = safe_scipy_stats()
	if mannwhitneyu is None:
		out["mann_whitney_u_label0_vs_label1"] = {"error": scipy_error}
		return out

	try:
		s0 = out["by_label"].get("0", {}).get("values", [])
		s1 = out["by_label"].get("1", {}).get("values", [])
		if s0 and s1:
			u_stat, p_val = mannwhitneyu(s0, s1, alternative="two-sided")
			out["mann_whitney_u_label0_vs_label1"] = {
				"u_stat": float(u_stat),
				"p_value": float(p_val),
				"mean_label0": float(np.mean(s0)),
				"mean_label1": float(np.mean(s1)),
				"median_label0": float(median(s0)),
				"median_label1": float(median(s1)),
				"hypothesis": "label0 distribution != label1 distribution",
			}
	except Exception as e:
		out["mann_whitney_u_label0_vs_label1"] = {"error": str(e)}

	return out


def build_records(raw_data):
	records = []
	invalid_date = 0
	for item in raw_data:
		claim = clean_text(item.get("claim", ""))
		parsed = parse_date(item.get("date"))
		if parsed is None:
			invalid_date += 1
		label = safe_int(item.get("label"), default=-1)

		evd_texts_all = get_evidence_texts(item)
		evd_texts_non_empty = [clean_text(t) for t in evd_texts_all if clean_text(t)]
		evidence_slots = len(evd_texts_non_empty)
		evidence_concat = " ".join(evd_texts_non_empty)

		records.append(
			{
				"id": str(item.get("id", "")),
				"claim": claim,
				"label": label,
				"domain": clean_text(item.get("domain", "")),
				"source": clean_text(item.get("source", "")),
				"source_desc": clean_text(item.get("source_desc", "")),
				"date": item.get("date"),
				"parsed_date": parsed,
				"claim_len": len(claim),
				"evidence_texts": evd_texts_non_empty,
				"evidence_lens": [len(t) for t in evd_texts_non_empty],
				"evidence_slots": evidence_slots,
				"evidence_total_len": len(evidence_concat),
				"evidence_concat": evidence_concat,
			}
		)
	return records, invalid_date


def time_and_length_stats(records, invalid_date):
	monthly = Counter()
	claim_lens = []
	evidence_lens = []
	evidence_total_lens = []

	for r in records:
		if r["parsed_date"] is not None:
			monthly[r["parsed_date"].strftime("%Y-%m")] += 1
		claim_lens.append(r["claim_len"])
		evidence_lens.extend(r["evidence_lens"])
		evidence_total_lens.append(r["evidence_total_len"])

	return {
		"claim_time_distribution_monthly": {
			k: int(v) for k, v in sorted(monthly.items())
		},
		"date_parse": {
			"invalid_date_count": int(invalid_date),
			"valid_date_count": int(len(records) - invalid_date),
		},
		"length_distribution": {
			"claim_len": {
				"summary": basic_summary_stats(claim_lens),
				"histogram": histogram(claim_lens, bins=20),
				"values": [int(v) for v in claim_lens],
			},
			"evidence_len_single": {
				"summary": basic_summary_stats(evidence_lens),
				"histogram": histogram(evidence_lens, bins=20),
				"values": [int(v) for v in evidence_lens],
			},
			"evidence_len_per_claim_total": {
				"summary": basic_summary_stats(evidence_total_lens),
				"histogram": histogram(evidence_total_lens, bins=20),
				"values": [int(v) for v in evidence_total_lens],
			},
		},
	}


def label_distribution(records):
	cnt = Counter(r["label"] for r in records)
	total = len(records)
	return {
		str(k): {
			"count": int(v),
			"ratio": float(v / total) if total else None,
		}
		for k, v in sorted(cnt.items())
	}


def source_distribution_stats(records):
	normalized = []
	for r in records:
		source = r.get("source", "")
		source_desc = r.get("source_desc", "")
		if source:
			normalized.append(source)
		elif source_desc:
			normalized.append(source_desc)
		else:
			normalized.append("UNKNOWN")
	counts = Counter(normalized)
	total = len(normalized)

	rows = []
	for source, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
		rows.append(
			{
				"source": source,
				"count": int(count),
				"ratio": float(count / total) if total else None,
			}
		)

	return {
		"total_samples": int(total),
		"unique_source_count": int(len(counts)),
		"unknown_source_count": int(counts.get("UNKNOWN", 0)),
		"distribution": rows,
	}


def run_analysis(data_path, stopwords_path, blocklist_path, output_path):
	raw = load_json(data_path)
	records, invalid_date = build_records(raw)
	stopwords = load_stopwords(stopwords_path)
	blocklist = load_keyword_blocklist(blocklist_path)
	evidence_slots = evidence_slot_stats(records)
	tfidf_stats = tfidf_similarity_stats(records)
	sentiment_stats = evidence_sentiment_stats(records)

	evidence_less = evidence_slots.get("significance", {}).get(
		"mann_whitney_u_label0_less_label1"
	)
	tfidf_greater = tfidf_stats.get("mann_whitney_u_label0_greater_label1")
	sentiment_diff = sentiment_stats.get("mann_whitney_u_label0_vs_label1")

	result = {
		"meta": {
			"input_data": os.path.abspath(data_path),
			"sample_size": len(records),
			"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		},
		"keyword_settings": {
			"stopwords_path": os.path.abspath(stopwords_path),
			"keyword_blocklist_path": os.path.abspath(blocklist_path),
			"keyword_blocklist_size": len(blocklist),
			"keyword_blocklist": sorted(list(blocklist)),
		},
		"label_distribution": label_distribution(records),
		"source_distribution": source_distribution_stats(records),
		"time_and_length": time_and_length_stats(records, invalid_date),
		"keywords_by_time_period": top_keywords_by_period(records, stopwords, blocklist, top_n=30),
		"distorted_claim_topics_label1": distorted_claim_topics(records, stopwords, blocklist, top_n=40),
		"evidence_slot_distribution": evidence_slots,
		"claim_evidence_tfidf_similarity": tfidf_stats,
		"evidence_sentiment": sentiment_stats,
		"label_group_differences": {
			"evidence_slots_label0_less_label1": evidence_less,
			"tfidf_similarity_label0_greater_label1": tfidf_greater,
			"evidence_sentiment_label0_vs_label1": sentiment_diff,
		},
	}

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(result, f, ensure_ascii=False, indent=2)

	return result


def print_summary_tables(result):
	kw = result.get("keyword_settings", {})
	blocklist = kw.get("keyword_blocklist", [])
	rows_kw = []
	for i, w in enumerate(blocklist[:30], start=1):
		rows_kw.append([i, w])
	print_table(
		title=f"关键词禁用列表（可自由补充） | 总数: {kw.get('keyword_blocklist_size', 0)}",
		headers=["No.", "Keyword"],
		rows=rows_kw,
	)
	print(f"可编辑文件: {kw.get('keyword_blocklist_path', 'NA')}")

	diff = result.get("label_group_differences", {})
	e_slots = diff.get("evidence_slots_label0_less_label1") or {}
	tfidf = diff.get("tfidf_similarity_label0_greater_label1") or {}
	sent = diff.get("evidence_sentiment_label0_vs_label1") or {}

	rows_test = [
		[
			"Evidence slots (label0 < label1)",
			fmt_num(e_slots.get("mean_label0")),
			fmt_num(e_slots.get("mean_label1")),
			fmt_num(e_slots.get("p_value"), 6),
			p_to_stars(e_slots.get("p_value")),
		],
		[
			"TF-IDF sim (label0 > label1)",
			fmt_num(tfidf.get("mean_label0")),
			fmt_num(tfidf.get("mean_label1")),
			fmt_num(tfidf.get("p_value"), 6),
			p_to_stars(tfidf.get("p_value")),
		],
		[
			"Evidence sentiment (label0 vs label1)",
			fmt_num(sent.get("mean_label0")),
			fmt_num(sent.get("mean_label1")),
			fmt_num(sent.get("p_value"), 6),
			p_to_stars(sent.get("p_value")),
		],
	]
	print_table(
		title="Label=0/1 分组差异检验汇总（Mann-Whitney U）",
		headers=["Metric", "Mean(label0)", "Mean(label1)", "p-value", "Sig."],
		rows=rows_test,
	)


def main():
	current_dir = os.path.dirname(os.path.abspath(__file__))
	default_data = os.path.abspath(os.path.join(current_dir, "..", "data", "health_info.json"))
	default_stopwords = os.path.abspath(os.path.join(current_dir, "..", "data", "stopwords_zh.txt"))
	default_blocklist = os.path.join(current_dir, "keyword_blocklist.txt")
	default_output = os.path.join(current_dir, "descriptive_stats.json")

	parser = argparse.ArgumentParser(description="Descriptive analysis for health_info.json")
	parser.add_argument("--data", default=default_data, help="Path to health_info.json")
	parser.add_argument("--stopwords", default=default_stopwords, help="Path to stopwords file")
	parser.add_argument(
		"--keyword-blocklist",
		default=default_blocklist,
		help="Path to keyword blocklist file (one keyword per line)",
	)
	parser.add_argument(
		"--output",
		default=default_output,
		help="Output json path (default: same directory as this script)",
	)
	args = parser.parse_args()

	result = run_analysis(args.data, args.stopwords, args.keyword_blocklist, args.output)
	print_summary_tables(result)
	print(f"Analysis finished. Samples={result['meta']['sample_size']}")
	print(f"JSON saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
	main()