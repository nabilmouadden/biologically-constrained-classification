# Per-abnormality accuracy of the released weights

GR-Neutro, 7 abnormality classes, DinoBloom-B backbone. Every number is from the
released run artifacts (`summary.json` for end-to-end models, `runs/residual_cbm/`
for the feature-bank heads). The two sections use different evaluation splits
(each end-to-end model's own held-out test split vs. the residual-CBM stratified
split), so compare within a section.

## End-to-end checkpoints (DinoBloom-B last-6 fine-tuned, seed 42)

### Overall

| Weight | Architecture | W-F1 | Macro-F1 | Subset acc. | Multi-seed mean W-F1 |
|---|---|--:|--:|--:|--:|
| joint_cbm_dinobloomB_ft_s42.pt | Joint CBM | 0.912 | 0.861 | 0.865 | 0.886 ± 0.014 |
| pure_bottleneck_cbm_dinobloomB_ft_s42.pt | Pure-bottleneck CBM (λ=2) | 0.909 | 0.862 | 0.847 | ≈ 0.88 |
| backbone_baseline_dinobloomB_ft_s42.pt | No-concept backbone baseline | 0.907 | 0.865 | 0.868 | 0.890 |
| cbm_sequential_dinobloomB_ft_s42.pt | Sequential CBM | 0.873 | 0.804 | 0.756 | — |
| cbm_independent_dinobloomB_ft_s42.pt | Independent CBM | 0.772 | 0.748 | 0.585 | — |

### Per-class F1

| Class (support) | Joint CBM | Pure-bottleneck | Backbone baseline | Sequential | Independent |
|---|--:|--:|--:|--:|--:|
| Normal (199) | 0.987 | 0.980 | 0.982 | 0.962 | 0.791 |
| Hypogranulation (108) | 0.903 | 0.910 | 0.897 | 0.888 | 0.824 |
| Hyposegmentation (67) | 0.855 | 0.826 | 0.821 | 0.846 | 0.786 |
| Chromatin (35) | 0.694 | 0.750 | 0.719 | 0.429 | 0.437 |
| Hypersegmentation (19) | 0.895 | 0.919 | 0.919 | 0.919 | 0.872 |
| Döhle (19) | 0.848 | 0.757 | 0.848 | 0.743 | 0.684 |
| Hypergranulation (16) | 0.842 | 0.889 | 0.865 | 0.842 | 0.842 |

## Feature-bank heads (frozen DinoBloom-B, 6-seed mean W-F1 [95% CI])

Reproduce from the released feature banks via `code/residual_cbm.py`.

| Method | W-F1 | Macro-F1 | Accuracy | Interpretable? |
|---|--:|--:|--:|---|
| backbone_mlp (reference) | 0.8495 [0.844, 0.855] | 0.771 | 0.852 | no |
| cem (Concept Embedding Model) | 0.8471 [0.838, 0.856] | 0.768 | 0.848 | yes |
| pcbmh (residual CBM, r=10) | 0.8297 [0.817, 0.842] | 0.735 | 0.830 | yes |
| pcbmh_highrank (r=64) | 0.8318 [0.828, 0.836] | 0.740 | 0.833 | yes |
| pure_bottleneck (fully transparent) | 0.7984 [0.788, 0.808] | 0.698 | 0.796 | yes |

Controls: PCBM-h vs matched-rank random-orthogonal residual = **+0.0385 W-F1
[+0.033, +0.044]** (concepts load-bearing for accuracy). Intervention faithfulness:
PCBM-h / CEM full 0→1 intervention gain = **+0.002 / −0.001 W-F1** (heads route
around the concepts — useful ≠ faithful).
