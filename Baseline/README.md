# 证据置换前baseline训练结果
============================================================
[Sigma] Final Distribution Report (5 Runs)
============================================================
Macro-F1 Avg: 0.7237 ± 0.0195
Micro-F1 Avg: 0.8196 ± 0.0159
------------------------------------------------------------
Category        Precision            Recall               F1-score
Real            0.7198 ± 0.0677      0.4728 ± 0.0799      0.5616 ± 0.0401     
Fake            0.8438 ± 0.0183      0.9334 ± 0.0344      0.8858 ± 0.0135     
macro avg       0.7818 ± 0.0286      0.7031 ± 0.0254      0.7237 ± 0.0195     
weighted avg    0.8130 ± 0.0149      0.8196 ± 0.0159      0.8056 ± 0.0158     
accuracy                                                  0.8196 ± 0.0159     
============================================================

# 证据置换后baseline训练结果
============================================================
[Sigma] Final Distribution Report (5 Runs)
============================================================
Macro-F1 Avg: 0.7386 ± 0.0160
Micro-F1 Avg: 0.8154 ± 0.0158
------------------------------------------------------------
Category        Precision            Recall               F1-score
Real            0.6153 ± 0.0505      0.5885 ± 0.0627      0.5974 ± 0.0259
Fake            0.8760 ± 0.0178      0.8849 ± 0.0288      0.8799 ± 0.0122
macro avg       0.7456 ± 0.0232      0.7367 ± 0.0223      0.7386 ± 0.0160
weighted avg    0.8153 ± 0.0158      0.8154 ± 0.0158      0.8139 ± 0.0144
accuracy                                                  0.8154 ± 0.0158
============================================================

# 图证据推理网络模型训练结果
Epoch 1/20  loss=0.5642  acc=0.8333  macro_f1=0.7784  mcc=0.5610
Epoch 2/20  loss=0.4296  acc=0.7864  macro_f1=0.7510  mcc=0.5552
Epoch 3/20  loss=0.3684  acc=0.8872  macro_f1=0.8352  mcc=0.6722
Epoch 4/20  loss=0.3390  acc=0.8453  macro_f1=0.8090  mcc=0.6434
Epoch 5/20  loss=0.3121  acc=0.8733  macro_f1=0.8385  mcc=0.6917
Epoch 6/20  loss=0.2776  acc=0.9012  macro_f1=0.8614  mcc=0.7228
Epoch 7/20  loss=0.2623  acc=0.8703  macro_f1=0.8345  mcc=0.6833
Epoch 8/20  loss=0.2485  acc=0.8812  macro_f1=0.8425  mcc=0.6902
Epoch 9/20  loss=0.2102  acc=0.9042  macro_f1=0.8637  mcc=0.7276
Epoch 10/20  loss=0.1872  acc=0.9042  macro_f1=0.8584  mcc=0.7199
Epoch 11/20  loss=0.1807  acc=0.8573  macro_f1=0.8210  mcc=0.6619
Epoch 12/20  loss=0.1380  acc=0.8862  macro_f1=0.8509  mcc=0.7093
Epoch 13/20  loss=0.1085  acc=0.8743  macro_f1=0.8388  mcc=0.6906
Epoch 14/20  loss=0.0924  acc=0.7465  macro_f1=0.7177  mcc=0.5240
Epoch 15/20  loss=0.0755  acc=0.8932  macro_f1=0.8565  mcc=0.7162
Epoch 16/20  loss=0.0472  acc=0.8862  macro_f1=0.8452  mcc=0.6920
Epoch 17/20  loss=0.0572  acc=0.8972  macro_f1=0.8579  mcc=0.7163
Epoch 18/20  loss=0.0355  acc=0.8912  macro_f1=0.8431  mcc=0.6869
Epoch 19/20  loss=0.0199  acc=0.8892  macro_f1=0.8441  mcc=0.6882
Epoch 20/20  loss=0.0198  acc=0.8942  macro_f1=0.8509  mcc=0.7018

# 消融实验结果
1. full model
  "acc": 0.8942115768463074,
  "macro_f1": 0.8509104384590767,
  "micro_f1": 0.8942115768463074,
  "mcc": 0.7018208769181532
  "classification_report": {
    "Real": {
      "precision": 0.7705627705627706,
      "recall": 0.7705627705627706,
      "f1-score": 0.7705627705627706,
      "support": 231.0
    },
    "Fake": {
      "precision": 0.9312581063553826,
      "recall": 0.9312581063553826,
      "f1-score": 0.9312581063553826,
      "support": 771.0
    }
  }
2. without TF-IDF
{
  "exp_name": "no_tfidf",
  "macro_f1_avg": 0.8459620825906491,
  "micro_f1_avg": 0.8892215568862275,
  "real_precision_avg": 0.75,
  "real_recall_avg": 0.7792207792207793,
  "real_f1_avg": 0.7643312101910829,
  "fake_precision_avg": 0.9330708661417323,
  "fake_recall_avg": 0.9221789883268483,
  "fake_f1_avg": 0.9275929549902152,
  "mcc_avg": 0.6921746504534586
}
3. without C-E NLI
{
  "exp_name": "no_ce",
  "macro_f1_avg": 0.8109472455427132,
  "micro_f1_avg": 0.8612774451097804,
  "real_precision_avg": 0.6811023622047244,
  "real_recall_avg": 0.7489177489177489,
  "real_f1_avg": 0.7134020618556701,
  "fake_precision_avg": 0.9224598930481284,
  "fake_recall_avg": 0.8949416342412452,
  "fake_f1_avg": 0.9084924292297564,
  "mcc_avg": 0.6233852912646826
}
4. without E-E NLI
{
  "exp_name": "no_ee",
  "macro_f1_avg": 0.8370053920862733,
  "micro_f1_avg": 0.8782435129740519,
  "real_precision_avg": 0.704119850187266,
  "real_recall_avg": 0.8138528138528138,
  "real_f1_avg": 0.7550200803212851,
  "fake_precision_avg": 0.9414965986394558,
  "fake_recall_avg": 0.8975356679636836,
  "fake_f1_avg": 0.9189907038512616,
  "mcc_avg": 0.6777050282878236
}
5. no feats
{
  "exp_name": "no_feats",
  "macro_f1_avg": 0.7917128603104213,
  "micro_f1_avg": 0.8383233532934131,
  "real_precision_avg": 0.6161616161616161,
  "real_recall_avg": 0.7922077922077922,
  "real_f1_avg": 0.6931818181818182,
  "fake_precision_avg": 0.9319148936170213,
  "fake_recall_avg": 0.8521400778210116,
  "fake_f1_avg": 0.8902439024390244,
  "mcc_avg": 0.5942658762613633
}
