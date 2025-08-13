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


python3 finetune_player_models.py --data_dir ./data --base_ckpt ./best_vit_amd.pth --max_tokens 256 --holdout_last_n 0 --ema
/mnt/2c4e8cd8-1388-4efb-9b39-77c162e4ff80/HavenLLM/experiments/experiments/finetune_player_models.py:422: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  base_raw = torch.load(args.base_ckpt, map_location="cpu")
/home/sysop/.local/lib/python3.10/site-packages/torch/nn/modules/transformer.py:307: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/mnt/2c4e8cd8-1388-4efb-9b39-77c162e4ff80/HavenLLM/experiments/experiments/finetune_player_models.py:249: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  base = torch.load(base_ckpt_path, map_location="cpu")
[INFO] Resized positional embedding: 33 -> 257
[INFO] Loaded base weights. Missing keys: 0, Unexpected keys: 0
/mnt/2c4e8cd8-1388-4efb-9b39-77c162e4ff80/HavenLLM/experiments/experiments/finetune_player_models.py:71: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  return torch.load(path, map_location="cpu")
/mnt/2c4e8cd8-1388-4efb-9b39-77c162e4ff80/HavenLLM/experiments/experiments/finetune_player_models.py:511: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
\n===== Fine‑tuning on training_all_AnishGiri  (N=40000 | holdout=0) =====
/mnt/2c4e8cd8-1388-4efb-9b39-77c162e4ff80/HavenLLM/experiments/experiments/finetune_player_models.py:313: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
Epoch 01 | train inf | val 166153489569594169829933683342049280.0000 | top1 0.4083 | time 48.0s
  ✔ Saved BEST to outputs_players/training_all_AnishGiri/training_all_AnishGiri_best.pth
Epoch 02 | train inf | val 166153489569594169829933683342049280.0000 | top1 0.4105 | time 47.5s
  ✔ Saved BEST to outputs_players/training_all_AnishGiri/training_all_AnishGiri_best.pth
Epoch 03 | train inf | val 166153489569594169829933683342049280.0000 | top1 0.4113 | time 47.5s
  ✔ Saved BEST to outputs_players/training_all_AnishGiri/training_all_AnishGiri_best.pth
Epoch 04 | train 5107.3364 | val 166153489569594169829933683342049280.0000 | top1 0.4090 | time 47.5s
Epoch 05 | train 5080.5248 | val 166153489569594169829933683342049280.0000 | top1 0.4103 | time 47.5s
Epoch 06 | train 5082.4794 | val 166153489569594169829933683342049280.0000 | top1 0.4068 | time 47.6s
Early stopping triggered.
\n===== Fine‑tuning on training_all_GMWSO  (N=40000 | holdout=0) =====
Epoch 01 | train 5078.5452 | val 83076744784797084914966841671024640.0000 | top1 0.4163 | time 47.8s
  ✔ Saved BEST to outputs_players/training_all_GMWSO/training_all_GMWSO_best.pth
Epoch 02 | train 5088.3735 | val 83076744784797084914966841671024640.0000 | top1 0.4190 | time 47.5s
  ✔ Saved BEST to outputs_players/training_all_GMWSO/training_all_GMWSO_best.pth
Epoch 03 | train 5079.2908 | val 83076744784797084914966841671024640.0000 | top1 0.4235 | time 47.5s
  ✔ Saved BEST to outputs_players/training_all_GMWSO/training_all_GMWSO_best.pth
Epoch 04 | train 5082.3768 | val 83076744784797084914966841671024640.0000 | top1 0.4253 | time 47.5s
  ✔ Saved BEST to outputs_players/training_all_GMWSO/training_all_GMWSO_best.pth
Epoch 05 | train 5084.8079 | val 83076744784797084914966841671024640.0000 | top1 0.4248 | time 47.5s
Epoch 06 | train 5058.5405 | val 83076744784797084914966841671024640.0000 | top1 0.4203 | time 47.5s
Epoch 07 | train 5078.2648 | val 83076744784797084914966841671024640.0000 | top1 0.4178 | time 47.5s
Early stopping triggered.
\n===== Fine‑tuning on training_all_GukeshDommaraju  (N=40000 | holdout=0) =====
Epoch 01 | train inf | val 581537213493579594404767891697172480.0000 | top1 0.4283 | time 47.9s
  ✔ Saved BEST to outputs_players/training_all_GukeshDommaraju/training_all_GukeshDommaraju_best.pth
Epoch 02 | train inf | val 581537213493579594404767891697172480.0000 | top1 0.4288 | time 47.5s
  ✔ Saved BEST to outputs_players/training_all_GukeshDommaraju/training_all_GukeshDommaraju_best.pth
Epoch 03 | train 5044.6310 | val 581537213493579594404767891697172480.0000 | top1 0.4288 | time 47.5s
Epoch 04 | train 5049.6408 | val 581537213493579594404767891697172480.0000 | top1 0.4275 | time 47.5s
Epoch 05 | train 5056.7561 | val 581537213493579594404767891697172480.0000 | top1 0.4298 | time 47.5s
  ✔ Saved BEST to outputs_players/training_all_GukeshDommaraju/training_all_GukeshDommaraju_best.pth
Epoch 06 | train 5069.5274 | val 581537213493579594404767891697172480.0000 | top1 0.4255 | time 47.5s
Epoch 07 | train 5023.9244 | val 581537213493579594404767891697172480.0000 | top1 0.4238 | time 47.5s
Epoch 08 | train 5071.2921 | val 581537213493579594404767891697172480.0000 | top1 0.4190 | time 47.5s
Early stopping triggered.
\n===== Fine‑tuning on training_all_MagnusCarlsen  (N=40000 | holdout=0) =====
Epoch 01 | train inf | val 166153489569594169829933683342049280.0000 | top1 0.3975 | time 47.8s
  ✔ Saved BEST to outputs_players/training_all_MagnusCarlsen/training_all_MagnusCarlsen_best.pth
Epoch 02 | train inf | val 166153489569594169829933683342049280.0000 | top1 0.4002 | time 47.6s
  ✔ Saved BEST to outputs_players/training_all_MagnusCarlsen/training_all_MagnusCarlsen_best.pth
Epoch 03 | train 5088.4833 | val 166153489569594169829933683342049280.0000 | top1 0.3995 | time 47.6s
Epoch 04 | train 5104.5166 | val 166153489569594169829933683342049280.0000 | top1 0.4005 | time 47.6s
  ✔ Saved BEST to outputs_players/training_all_MagnusCarlsen/training_all_MagnusCarlsen_best.pth
Epoch 05 | train 5086.0944 | val 166153489569594169829933683342049280.0000 | top1 0.3992 | time 47.6s
Epoch 06 | train 5076.4352 | val 166153489569594169829933683342049280.0000 | top1 0.3985 | time 47.6s
Epoch 07 | train 5111.9316 | val 166153489569594169829933683342049280.0000 | top1 0.3970 | time 47.6s
Early stopping triggered.
\n===== Fine‑tuning on training_all_rpragchess  (N=40000 | holdout=0) =====
Epoch 01 | train inf | val 83076744784797084914966841671024640.0000 | top1 0.3757 | time 47.8s
  ✔ Saved BEST to outputs_players/training_all_rpragchess/training_all_rpragchess_best.pth
Epoch 02 | train inf | val 83076744784797084914966841671024640.0000 | top1 0.3762 | time 47.6s
  ✔ Saved BEST to outputs_players/training_all_rpragchess/training_all_rpragchess_best.pth
Epoch 03 | train inf | val 83076744784797084914966841671024640.0000 | top1 0.3777 | time 47.5s
  ✔ Saved BEST to outputs_players/training_all_rpragchess/training_all_rpragchess_best.pth
Epoch 04 | train 5129.4410 | val 83076744784797084914966841671024640.0000 | top1 0.3745 | time 47.6s
Epoch 05 | train inf | val 83076744784797084914966841671024640.0000 | top1 0.3735 | time 47.6s
Epoch 06 | train inf | val 83076744784797084914966841671024640.0000 | top1 0.3690 | time 47.7s
Early stopping triggered.
\nAll players done.
Epoch 04 | train 5058.9054 | val 401859602679948681241353990543769600.0000 | top1 0.3793 | time 130.0s
Early stopping triggered.
\nAll players done.
