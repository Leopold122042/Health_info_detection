import time
import tracemalloc
from threading import Event, Thread
from pathlib import Path

import numpy as np

from milvus_reuse_pipeline import EvidenceReusePipeline

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


class ProcessMemorySampler:
    def __init__(self, interval_sec=0.02):
        self.interval_sec = interval_sec
        self._stop_event = Event()
        self._thread = None
        self._peak_rss_bytes = 0
        self._start_rss_bytes = None
        self._end_rss_bytes = None

    def start(self):
        if psutil is None:
            return

        process = psutil.Process()
        self._start_rss_bytes = process.memory_info().rss

        def _sample_loop():
            while not self._stop_event.is_set():
                try:
                    rss = process.memory_info().rss
                    if rss > self._peak_rss_bytes:
                        self._peak_rss_bytes = rss
                except psutil.Error:
                    pass
                time.sleep(self.interval_sec)

        self._thread = Thread(target=_sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if psutil is None:
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)

        try:
            process = psutil.Process()
            self._end_rss_bytes = process.memory_info().rss
            self._peak_rss_bytes = max(self._peak_rss_bytes, self._end_rss_bytes)
        except psutil.Error:
            pass

    @staticmethod
    def _to_mb(value):
        if value is None:
            return None
        return float(value / (1024 ** 2))

    def snapshot(self):
        return {
            "rss_start_mb": self._to_mb(self._start_rss_bytes),
            "rss_end_mb": self._to_mb(self._end_rss_bytes),
            "rss_peak_mb": self._to_mb(self._peak_rss_bytes if self._peak_rss_bytes > 0 else None),
        }


class CacheRetrievalBenchmark:
    def __init__(
        self,
        encoder,
        cache_dir,
        sample_size=1000,
        topk=5,
        nprobe=16,
        nlist=128,
        collection_prefix="evidence_pool_benchmark",
    ):
        self.encoder = encoder
        self.cache_dir = Path(cache_dir)
        self.sample_size = int(sample_size)
        self.topk = int(topk)
        self.nprobe = int(nprobe)
        self.nlist = int(nlist)
        self.collection_prefix = collection_prefix

    def run(self, claim_texts):
        claim_texts_sample = claim_texts[: self.sample_size]
        actual_n = len(claim_texts_sample)

        if actual_n == 0:
            return {
                "config": {
                    "requested_claim_sample_size": self.sample_size,
                    "actual_claim_sample_size": 0,
                    "topk": self.topk,
                    "nprobe": self.nprobe,
                    "groups": ["A", "C"],
                },
                "groups": {},
                "comparison": {"note": "No claim samples available."},
            }

        group_a = self._safe_run_group("A", lambda: self._run_group_a(claim_texts_sample, actual_n))
        group_c = self._safe_run_group("C", lambda: self._run_group_c(actual_n))

        return {
            "config": {
                "requested_claim_sample_size": self.sample_size,
                "actual_claim_sample_size": actual_n,
                "topk": self.topk,
                "nprobe": self.nprobe,
                "groups": ["A", "C"],
                "group_desc": {
                    "A": "Evidence与Claim都在线编码（无.npy缓存）",
                    "C": "Evidence与Claim都从.npy加载",
                },
            },
            "groups": {
                "A": group_a,
                "C": group_c,
            },
            "comparison": self._build_comparison(group_a, group_c),
        }

    @staticmethod
    def _safe_run_group(group_name, fn):
        try:
            result = fn()
            result["status"] = "ok"
            return result
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"{group_name} benchmark failed: {exc}",
            }

    def _load_evidence_embeddings(self):
        start = time.perf_counter()
        evidence_path = self.cache_dir / "evidences_embeddings_prev.npy"
        evidence_embeddings = np.load(evidence_path)

        if evidence_embeddings.ndim == 3:
            mask_path = self.cache_dir / "evd_mask_prev.npy"
            if mask_path.exists():
                evd_mask = np.load(mask_path)
                valid_indices = np.where(evd_mask == 1)
                evidence_embeddings = evidence_embeddings[valid_indices]
            else:
                evidence_embeddings = evidence_embeddings.reshape(-1, evidence_embeddings.shape[-1])

        elapsed = time.perf_counter() - start
        return evidence_embeddings, elapsed

    def _load_claim_embeddings(self, sample_size):
        start = time.perf_counter()
        claims_path = self.cache_dir / "claims_embeddings.npy"
        claim_embeddings = np.load(claims_path)[:sample_size]
        elapsed = time.perf_counter() - start
        return claim_embeddings, elapsed

    def _run_group_a(self, claim_texts_sample, actual_n):
        evidence_texts = self._load_evidence_texts()

        start_encode_evidence = time.perf_counter()
        evidence_embeddings = self.encoder.encode(
            evidence_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        t_encode_evidence = time.perf_counter() - start_encode_evidence

        start_encode = time.perf_counter()
        claim_embeddings = self.encoder.encode(
            claim_texts_sample,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        t_encode_claims = time.perf_counter() - start_encode

        payload = self._run_retrieval_pipeline(
            claim_embeddings=claim_embeddings,
            evidence_embeddings=evidence_embeddings,
            group_name="A",
            actual_n=actual_n,
            extra_stage_times={
                "load_evidence_cache_sec": 0.0,
                "encode_evidence_online_sec": t_encode_evidence,
                "encode_claims_online_sec": t_encode_claims,
                "load_claim_cache_sec": 0.0,
            },
        )
        return payload

    def _run_group_c(self, actual_n):
        evidence_embeddings, t_load_evidence = self._load_evidence_embeddings()
        claim_embeddings, t_load_claim = self._load_claim_embeddings(actual_n)

        payload = self._run_retrieval_pipeline(
            claim_embeddings=claim_embeddings,
            evidence_embeddings=evidence_embeddings,
            group_name="C",
            actual_n=actual_n,
            extra_stage_times={
                "load_evidence_cache_sec": t_load_evidence,
                "encode_evidence_online_sec": 0.0,
                "encode_claims_online_sec": 0.0,
                "load_claim_cache_sec": t_load_claim,
            },
        )
        return payload

    @staticmethod
    def _load_evidence_texts():
        import json

        data_path = Path("data/health_info.json")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        evidence_texts = []
        for item in data:
            for ev in item.get("evidence", {}).values():
                content = ev.get("content", "").strip()
                if content:
                    evidence_texts.append(content)
        return evidence_texts

    def _run_retrieval_pipeline(
        self,
        claim_embeddings,
        evidence_embeddings,
        group_name,
        actual_n,
        extra_stage_times,
    ):
        sampler = ProcessMemorySampler(interval_sec=0.02)

        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_available = True
        else:
            gpu_available = False

        tracemalloc.start()
        cpu_time_start = time.process_time()
        wall_start = time.perf_counter()
        sampler.start()

        pipeline = None
        try:
            t0 = time.perf_counter()
            pipeline = EvidenceReusePipeline(
                dim=claim_embeddings.shape[1],
                collection_name=f"{self.collection_prefix}_{group_name.lower()}_{int(time.time() * 1000)}",
            )
            pipeline.build_collection()
            t_collection = time.perf_counter() - t0

            t1 = time.perf_counter()
            pipeline.insert_evidences(evidence_embeddings)
            t_insert = time.perf_counter() - t1

            t2 = time.perf_counter()
            pipeline.build_index(nlist=self.nlist)
            t_index = time.perf_counter() - t2

            t3 = time.perf_counter()
            _ = pipeline.retrieve(claim_embeddings, topk=self.topk, nprobe=self.nprobe)
            t_search = time.perf_counter() - t3

        finally:
            sampler.stop()
            total_wall = time.perf_counter() - wall_start
            total_cpu = time.process_time() - cpu_time_start
            py_current, py_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if pipeline is not None:
                try:
                    pipeline.collection.release()
                except Exception:
                    pass

        gpu_peak_mb = None
        if gpu_available:
            gpu_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

        stage_times = {
            **extra_stage_times,
            "build_collection_sec": float(t_collection),
            "insert_evidences_sec": float(t_insert),
            "build_index_sec": float(t_index),
            "search_sec": float(t_search),
        }

        retrieval_total = (
            stage_times["build_collection_sec"]
            + stage_times["insert_evidences_sec"]
            + stage_times["build_index_sec"]
            + stage_times["search_sec"]
        )

        return {
            "claim_count": int(actual_n),
            "latency": {
                "total_wall_sec": float(total_wall),
                "per_claim_total_ms": float(total_wall * 1000.0 / max(actual_n, 1)),
                "search_sec": float(t_search),
                "per_claim_search_ms": float(t_search * 1000.0 / max(actual_n, 1)),
                "retrieval_pipeline_sec": float(retrieval_total),
            },
            "throughput": {
                "claims_per_sec_total": float(actual_n / total_wall) if total_wall > 0 else None,
                "claims_per_sec_search": float(actual_n / t_search) if t_search > 0 else None,
            },
            "resources": {
                "cpu_process_time_sec": float(total_cpu),
                "python_peak_alloc_mb": float(py_peak / (1024 ** 2)),
                "gpu_peak_alloc_mb": gpu_peak_mb,
                **sampler.snapshot(),
            },
            "stage_time_breakdown_sec": stage_times,
        }

    @staticmethod
    def _ratio(old_value, new_value):
        if old_value is None or new_value is None or old_value == 0:
            return None
        return float((old_value - new_value) / old_value)

    def _build_comparison(self, group_a, group_c):
        if group_a.get("status") != "ok" or group_c.get("status") != "ok":
            return {
                "c_vs_a": {
                    "total_wall_time_reduction_ratio": None,
                    "per_claim_latency_reduction_ratio": None,
                    "total_throughput_improvement_ratio": None,
                    "note": "A或C组执行失败，无法比较",
                }
            }

        a_total = group_a["latency"]["total_wall_sec"]
        c_total = group_c["latency"]["total_wall_sec"]
        a_per_claim = group_a["latency"]["per_claim_total_ms"]
        c_per_claim = group_c["latency"]["per_claim_total_ms"]
        a_tps = group_a["throughput"]["claims_per_sec_total"]
        c_tps = group_c["throughput"]["claims_per_sec_total"]

        throughput_gain = None
        if a_tps and a_tps > 0 and c_tps is not None:
            throughput_gain = float((c_tps - a_tps) / a_tps)

        return {
            "c_vs_a": {
                "total_wall_time_reduction_ratio": self._ratio(a_total, c_total),
                "per_claim_latency_reduction_ratio": self._ratio(a_per_claim, c_per_claim),
                "total_throughput_improvement_ratio": throughput_gain,
                "note": "正值表示C组优于A组",
            }
        }