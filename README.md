\n===== Fine‑tuning on Carlsen_Magnus  (N=162221 | holdout=20000) =====
/mnt/2c4e8cd8-1388-4efb-9b39-77c162e4ff80/HavenLLM/experiments/experiments/finetune_player_models.py:313: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
Epoch 01 | train inf | val inf | top1 0.4212 | time 170.6s
Epoch 02 | train inf | val inf | top1 0.4126 | time 170.3s
Epoch 03 | train 5092.1497 | val inf | top1 0.3943 | time 170.4s
Early stopping triggered.
\n===== Fine‑tuning on Caruana_Fabiano  (N=149713 | holdout=20000) =====
Epoch 01 | train inf | val inf | top1 0.4135 | time 155.9s
Epoch 02 | train 5062.7931 | val inf | top1 0.4080 | time 155.7s
Epoch 03 | train 5074.6874 | val inf | top1 0.3982 | time 155.7s
Early stopping triggered.
\n===== Fine‑tuning on Ding_Liren  (N=65123 | holdout=20000) =====
Epoch 01 | train inf | val inf | top1 0.4011 | time 54.4s
Epoch 02 | train inf | val inf | top1 0.4006 | time 54.2s
Epoch 03 | train inf | val inf | top1 0.4002 | time 54.2s
Early stopping triggered.
\n===== Fine‑tuning on Gelfand_Boris  (N=159872 | holdout=20000) =====
Epoch 01 | train inf | val inf | top1 0.4234 | time 167.9s
Epoch 02 | train inf | val inf | top1 0.4164 | time 167.7s
Epoch 03 | train 5091.9657 | val inf | top1 0.4076 | time 167.7s
Early stopping triggered.
\n===== Fine‑tuning on Grischuk_Alexander  (N=157061 | holdout=20000) =====
Epoch 01 | train inf | val inf | top1 0.4393 | time 164.5s
Epoch 02 | train inf | val inf | top1 0.4328 | time 164.3s
Epoch 03 | train 5046.9626 | val inf | top1 0.4190 | time 164.3s
Early stopping triggered.
\n===== Fine‑tuning on Ivanchuk_Vasyl  (N=197170 | holdout=20000) =====
Epoch 01 | train inf | val inf | top1 0.4219 | time 212.7s
Epoch 02 | train 5072.5321 | val inf | top1 0.4114 | time 212.5s
Epoch 03 | train 5073.7113 | val inf | top1 0.3987 | time 212.5s
Early stopping triggered.
\n===== Fine‑tuning on Karpov_Anatoly  (N=167265 | holdout=20000) =====
Epoch 01 | train inf | val inf | top1 0.4139 | time 177.1s
Epoch 02 | train inf | val inf | top1 0.4046 | time 176.7s
Epoch 03 | train 5158.4401 | val inf | top1 0.3929 | time 176.7s
Early stopping triggered.
\n===== Fine‑tuning on Kasparov_Garry  (N=77845 | holdout=20000) =====
Epoch 01 | train inf | val 1155850362223263802953795413655683072.0000 | top1 0.4403 | time 69.9s
  ✔ Saved BEST to outputs_players/Kasparov_Garry/Kasparov_Garry_best.pth
Epoch 02 | train inf | val 1155850362223263802953795413655683072.0000 | top1 0.4432 | time 69.6s
  ✔ Saved BEST to outputs_players/Kasparov_Garry/Kasparov_Garry_best.pth
Epoch 03 | train 5050.5482 | val 1155850362223263802953795413655683072.0000 | top1 0.4436 | time 69.6s
  ✔ Saved BEST to outputs_players/Kasparov_Garry/Kasparov_Garry_best.pth
Epoch 04 | train 5066.5134 | val 1155850362223263802953795413655683072.0000 | top1 0.4391 | time 69.6s
Epoch 05 | train 5072.7427 | val 1155850362223263802953795413655683072.0000 | top1 0.4322 | time 69.6s
Epoch 06 | train 5084.0152 | val 1155850362223263802953795413655683072.0000 | top1 0.4268 | time 69.6s
Early stopping triggered.
\n===== Fine‑tuning on Korchnoi_Viktor  (N=229543 | holdout=20000) =====
Epoch 01 | train inf | val inf | top1 0.3942 | time 251.8s
Epoch 02 | train inf | val inf | top1 0.3779 | time 251.4s
Epoch 03 | train 5074.6539 | val inf | top1 0.3671 | time 251.6s
Early stopping triggered.
\n===== Fine‑tuning on Kramnik_Vladimir  (N=140949 | holdout=20000) =====
Epoch 01 | train inf | val inf | top1 0.4018 | time 146.5s
Epoch 02 | train 5093.2493 | val inf | top1 0.3978 | time 145.3s
Epoch 03 | train 5106.4117 | val inf | top1 0.3850 | time 145.3s
Early stopping triggered.
\n===== Fine‑tuning on Nepomniachtchi_Ian  (N=128263 | holdout=20000) =====
Epoch 01 | train inf | val 401859602679948681241353990543769600.0000 | top1 0.3986 | time 130.4s
  ✔ Saved BEST to outputs_players/Nepomniachtchi_Ian/Nepomniachtchi_Ian_best.pth
Epoch 02 | train 5066.6532 | val 401859602679948681241353990543769600.0000 | top1 0.3941 | time 130.0s
Epoch 03 | train 5082.1629 | val 401859602679948681241353990543769600.0000 | top1 0.3847 | time 129.9s
Epoch 04 | train 5058.9054 | val 401859602679948681241353990543769600.0000 | top1 0.3793 | time 130.0s
Early stopping triggered.
\nAll players done.
