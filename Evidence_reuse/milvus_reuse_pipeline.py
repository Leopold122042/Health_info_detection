# milvus_reuse_pipeline.py
import faiss
from collections import defaultdict
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)

class EvidenceReusePipeline:
    def __init__(self, dim=384, collection_name="evidence_pool"):
        self.dim = dim
        self.collection_name = collection_name
        connections.connect("default", host="localhost", port="19530")

    def build_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(
                name="evidence_id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim
            )
        ]

        schema = CollectionSchema(
            fields,
            description="Evidence embeddings for reuse feasibility study"
        )

        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

    def insert_evidences(self, evidence_embeddings):
        faiss.normalize_L2(evidence_embeddings)
        ids = list(range(len(evidence_embeddings)))

        self.collection.insert([
            ids,
            evidence_embeddings.tolist()
        ])
        self.collection.flush()

    def build_index(self, nlist=128):
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": nlist}
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        self.collection.load()

    def retrieve(self, claim_embeddings, topk=5, nprobe=16):
        faiss.normalize_L2(claim_embeddings)

        results = self.collection.search(
            data=claim_embeddings.tolist(),
            anns_field="embedding",
            param={
                "metric_type": "IP",
                "params": {"nprobe": nprobe}
            },
            limit=topk,
            output_fields=["evidence_id"]
        )

        claim_to_evidence = {
            cid: [hit.id for hit in hits]
            for cid, hits in enumerate(results)
        }
        return claim_to_evidence

    @staticmethod
    def invert_mapping(claim_to_evidence):
        evidence_to_claims = defaultdict(list)
        for cid, ev_ids in claim_to_evidence.items():
            for ev_id in ev_ids:
                evidence_to_claims[ev_id].append(cid)
        return evidence_to_claims