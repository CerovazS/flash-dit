# flash-dit — TODO

## Bloccanti (necessari per il primo training)

- [ ] **Budget SLURM** — `iscrc_lens` ha budget esaurito/scaduto. Verificare con CINECA
      o sottomettere su `iscrc_yendri` (già testato funzionante).
      Aggiornare `slurm/*.sbatch` con l'account corretto.

- [ ] **Precompute latents** — Eseguire `sbatch slurm/precompute_latents.sbatch`.
      Output: `/leonardo_scratch/large/userexternal/lcerovaz/flash-dit-latents/stable_audio_open/fma_medium.h5`
      Stima: ~3-4h su 1 A100, ~16 GB HDF5.
      Verificare: shape `(N, 64, 64)` float16, attrs `mean/std` presenti.

## Training

- [ ] **Baseline 1 — vanilla_sit** — `RUN_NAME=vanilla_sit_001 MODEL=vanilla_sit sbatch slurm/train_single.sbatch`
- [ ] **Baseline 2 — flash_dit** — `RUN_NAME=flash_dit_001 MODEL=flash_dit sbatch slurm/train_single.sbatch`
- [ ] **Baseline 3 — flash_dit_recurse** — `RUN_NAME=flash_dit_recurse_001 MODEL=flash_dit_recurse sbatch slurm/train_single.sbatch`

## Validazione e metriche

- [ ] **FAD evaluation** — Una volta generati i sample WAV:
      `uv run python scripts/eval_fad.py --generated outputs/... --reference $FMA_VAL_WAVS`
- [ ] **Audio generation script** — Test end-to-end di `scripts/sample.py` con checkpoint reale.
      Verificare che il VAE decoder funzioni (`decode_audio`) e che i WAV siano corretti.

## Cose mancanti / da migliorare

- [ ] **Git identity** — Configurare `git config --global user.name` e `user.email`
      (ora usa nome/email automatici del cluster).
- [ ] **WandB / logging opzionale** — Il codice ha `use_wandb: false` in config ma
      non è ancora cablato in `scripts/train.py`. Aggiungere se necessario.
- [ ] **Smoke test senza GPU** — `scripts/train.py` non è testato end-to-end.
      Aggiungere un test `trainer.fast_dev_run=true` con latenti mock.
- [ ] **scripts/train.py proxy** — Quando il training carica il VAE da HF al volo,
      serve il proxy su compute node. Verificare che il proxy sia settato nel sbatch.
- [ ] **Differential Attention variant** (`flash_dit_diff.yaml`) — Prevista nel piano
      ma non ancora implementata in `attention.py`.
- [ ] **FSDP2 strategy** — Prevista nel piano per modelli Base+, non ancora implementata.
- [ ] **FA3 su H100** — Il flag `use_fa3=True` è nel codice ma FA3 non è installato
      su CINECA (solo A100). Testare su RunPod H100 (`root@103.207.149.120:14754`).
- [ ] **Flywheel logging** — Dopo il primo run completato, loggare su Flywheel
      con `/flywheel-log` (nodi empirical per ogni baseline).

## Note operative

- Proxy funzionante: `http://login01:1909`
- Account funzionante: `iscrc_yendri` (da verificare quale usare per questo progetto)
- HF cache: `/leonardo_scratch/fast/IscrC_LENS/lcerovaz/flash-dit-cache/hf_cache`
- Latents target: `/leonardo_scratch/large/userexternal/lcerovaz/flash-dit-latents/stable_audio_open/fma_medium.h5`
