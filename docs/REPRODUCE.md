# Reproducing the thesis results

Every figure, table and analysis in the thesis, with the script that produces it.

All paths are relative to `external/tokenhmr/`, and all commands assume the `thesis-HMR`
environment (see the main [README](../README.md)):

```bash
conda activate thesis-HMR
cd external/tokenhmr
```

Offscreen mesh rendering needs `PYOPENGL_PLATFORM=egl`.

## Order of work

1. **Stage 1**, train the tokenizers (`tokenization/train_poseVQ.py`). The ablation needs all
   eight; everything downstream needs only the FSQ `d = 4` one.
2. **Tokenizer analysis** (§4.2, §4.3). Runs on CPU against stage-1 checkpoints.
3. **Stage 2**, train the downstream models (`tokenhmr/train.py`). This is the expensive
   part: the from-scratch arms are 200k steps, roughly 14 h each on one GPU.
4. **Downstream analysis and figures** (§4.4).

The analysis scripts read stage-1 checkpoints and stage-2 `results/release/<run>/` outputs,
so steps 2 and 4 only re-run if the trained artifacts already exist.

## Stage 1: tokenizers

```bash
cd tokenization
python train_poseVQ.py --cfg configs/<config>.yaml
```

Outputs land in `output/<experiment_name>/`, with `best_net.pth` (best weights) and
`latest_checkpoint.pth` (full resumable state). The config-to-ablation-row mapping is in the
main README.

Validation writes `results_<VALLIST>.pkl`, holding the ground-truth poses used by
`calculate_codebook_utilization.py`. It is not checked in (~89 MB); regenerate it with a
validation pass if you need it.

## Tokenizer ablation and latent-space analysis

| Thesis artifact | Script | Notes |
| --- | --- | --- |
| Tab. 4.3 stability `S(δ)`, `Δ_NN` | `tokenization/analyze_token_stability.py` | writes `output/token_stability/summary.json`, which later scripts read for checkpoint paths |
| Tab. 4.3 usage | `tokenization/calculate_codebook_utilization.py` | needs `--pkl` and `--ckpt`; also reported by the geometry script below |
| Tab. 4.4 codebook geometry | `tokenization/analyze_codebook_geometry.py` | 8 tokenizers, same 2048 val poses / seed 0 as Tab. 4.3. CPU |
| Tab. 4.5 reconstruction by dataset | `tokenization/analyze_recon_by_dataset.py` | writes `output/recon_by_dataset/summary_all.json` |
| Fig. 4.3 recon/stability frontier | `thesis_figures/plot_recon_frontier.py` | numbers hard-coded from Tab. 4.3; edit the `P` list if a cell changes |
| Fig. `cb-separation` | `tokenization/dump_code_separation.py` → `thesis_figures/plot_code_separation.py` | dump writes `output/codebook_geometry/separation_arrays.npz` |
| Fig. 4.5 + Fig. B.1 influence maps | `tokenization/analyze_token_maps.py` → `thesis_figures/plot_influence_maps.py` | analysis writes `output/token_maps/<label>.npz` |
| Fig. B.3 latent scatter | `tokenization/analyze_token_maps.py` → `thesis_figures/plot_latent_scatter.py` | same `.npz` |
| Fig. 4.7 + Fig. B.2 token displacement | `tokenization/render_token_viz.py` | pyrender EGL render on GPU, decode on CPU |
| Fig. `app-token-map` (160 positions) | `tokenization/visualize_token_effects.py` → `thesis_figures/plot_token_map_page.py` | the page script re-tiles panels the first already rendered; no GPU |
| §4.3 exploratory single-run versions | `tokenization/analyze_codebook.py`, `visualize_latent_space.py`, `analyze_latent_pose_info.py` | superseded by the multi-tokenizer drivers above, kept for the single-run analyses |

## Stage 2: downstream models

```bash
python tokenhmr/train.py experiment=<name>
```

Each experiment config freezes a stage-1 tokenizer as its pose head through
`TOKENIZER_CHECKPOINT_PATH`; repoint it at your own output directory. The runs behind the
main downstream tables:

| Thesis artifact | Experiment config | Run name |
| --- | --- | --- |
| Tab. 4.9 row 1, token CE only, no gate | `tokenhmr_additive` (phase A) | `chain_nogate` |
| Tab. 4.9 row 2, `+` label-purity gate | `tokenhmr_additive` (phase B) | `chain_gate` |
| Tab. 4.9 row 3, `+` straight-through | `tokenhmr_additive` (phase D) | `chain_st` |
| Tab. 4.9 row 4, `+` Gumbel sampling | `tokenhmr_additive` (phase E) | `chain_gumbel` |
| Tab. 4.11 decoder-aware targets | `tokenhmr_decoder_aware` | not yet measured |
| Pose-supervised baselines | `tokenhmr_fsq`, `tokenhmr_transformer_*` | see `results/release/` |

`run_additive_queue.sh` runs the four chain arms end to end and evaluates each as it
finishes. Evaluation writes `results/release/<run>_{hard,soft}/eval_regression.csv`, with
`(hard_|soft_)mode_re`, `mode_mpjpe` and `mode_pve` for PA-MPJPE, MPJPE and PVE, plus
`token_top1_acc`, `token_top5_acc` and `token_pred_entropy`.

## Downstream analysis and figures

| Thesis artifact | Script | Notes |
| --- | --- | --- |
| Fig. `token-ambiguity` | `tokenhmr/analyze_token_ambiguity.py` → `thesis_figures/plot_token_ambiguity.py` | analysis dumps `results/ambiguity_gate/damages.npz` |
| Fig. `decoding-mechanism` | `thesis_figures/plot_decoding_figure.py` | caches intermediates in `thesis_figures/cache/` |
| Qualitative decode comparison | `thesis_figures/render_decode_comparison.py` → `make_decode_figure.py` | first renders a browsing set with a `ranking.csv`, then lays the chosen frames out |
| Qualitative no-CE vs CE | `thesis_figures/render_val_comparison.py` | COCO-VAL frames, comparable to the wandb `val/predictions` panels |
| Fig. `additive-chain` | `thesis_figures/plot_additive_chain.py` | figure no longer used in the thesis, but its `STEPS` list is the maintained source of Tab. 4.9's numbers |

## Methods figures

| Thesis artifact | Script | Output |
| --- | --- | --- |
| Fig. 3.1 SMPL decomposition | `thesis_figures/render_smpl_figure.py` | `images/smpl/smpl_decomposition.png` (uses Blender / `bpy`) |
| Fig. 3.8 skeleton attention mask | `thesis_figures/gen_skeleton_mask.py` | `images/tokenizer/skeleton_mesh.png` and the inline TikZ in `skeleton_mask_tikz.tex` |

## Interactive tools

Not thesis artifacts, but useful for inspecting a tokenizer:

- `tokenization/interactive_token_viewer.py` — web page to replace a single token and watch
  the decoded body change.
- `tokenhmr/visualize_video_latents.py` — side-by-side video of the recovered mesh and the
  live FSQ codes.

## Known gaps

- Tab. 4.10 (gated cross-entropy across four tokenizers): three of four rows are still
  projected rather than measured. Making them real needs those tokenizers re-run under the
  current chain recipe.
- Tab. 4.11 (decoder-aware targets): both rows not yet measured.
- The continuous-regression row of the downstream baseline table is deliberately blank.
- Fig. `app-qualitative`: placeholder, not yet produced.
