import torch
import torch.nn as nn
import torch.nn.functional as F

class ConflictAwareLayer(nn.Module):
    def __init__(self, embed_dim=768, use_nli=True, use_tfidf=True, use_ee=True):
        super().__init__()
        self.use_nli = use_nli
        self.use_tfidf = use_tfidf
        self.use_ee = use_ee
        self.node_proj = nn.Linear(embed_dim, embed_dim)
        # Evidence feature -> attention logit
        self.attn_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, claim, evidences, nli_ce, nli_ee, tfidf, mask, evd_feats=None):
        if evd_feats is None:
            # Fallback to uniform attention if features are missing
            attn_logits = torch.ones_like(mask).unsqueeze(-1)
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        else:
            feats = evd_feats.clone()
            # Feature order: [relevance, entail_ce, contra_ce, ee_support, ee_conflict]
            if not self.use_tfidf:
                feats[..., 0] = 0
            if not self.use_nli:
                feats[..., 1:3] = 0
            if not self.use_ee:
                feats[..., 3:5] = 0
            attn_logits = self.attn_mlp(feats)
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        attn_weights = F.softmax(attn_logits, dim=1)
        weighted_evid = torch.sum(evidences * attn_weights, dim=1)  # (B, 768)
        return weighted_evid, attn_weights

class SCAGNN(nn.Module):
    def __init__(self, embed_dim=768, num_classes=2, **model_params):
        super().__init__()
        self.graph_layer = ConflictAwareLayer(embed_dim, **model_params)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, batch):
        c = batch['claim']
        e = batch['evidences']
        
        # 通过图层得到聚合后的证据特征
        agg_evid, attn_dist = self.graph_layer(
            c, e, batch['nli_ce'], batch['nli_ee'], batch['tfidf'], batch['mask'], batch.get('evd_feats')
        )
        
        # 拼接声明特征与聚合证据特征进行分类
        combined = torch.cat([c, agg_evid], dim=-1)
        logits = self.classifier(combined)
        
        return logits, attn_dist
