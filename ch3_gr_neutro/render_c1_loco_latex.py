"""Render the honest LOCO LaTeX block for C1'.

Inputs: outputs/c1_loco/results.json (from eval_loco.py).
Output: papers/ai_conference/outputs/c1_loco/c1_loco_block.tex
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def fmt(d, key):
    if d is None or d.get(key) is None: return "--"
    blob = d[key]
    if blob is None: return "--"
    m = blob.get("mean"); s = blob.get("sd")
    if m is None: return "--"
    return f"{m:.3f}\\,$\\pm$\\,{s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/c1_loco/results.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blob = json.loads(Path(args.in_path).read_text())
    cfgs = blob["configs"]
    pub = blob["published_comparators"]

    def row(c):
        a = cfgs[c].get("aggregate", {})
        held = cfgs[c]["held_out_cohort"]
        return (f"\\textbf{{{c}}} & {held} & "
                f"{fmt(a, 'held_normal_recall')} & "
                f"{fmt(a, 'pooled_indist_wf1')} & "
                f"{fmt(a, 'pooled_indist_mf1')} & "
                f"{fmt(a, 'n_concepts_above_0p05')} & "
                f"{fmt(a, 'gap_sigma7_sigma8')} \\\\")

    rows = "\n".join(row(c) for c in ["A","B","C","D"] if c in cfgs)

    tex = r"""%% C1' --- LOCO-honest multi-cohort iVAE evaluation.
%% Replaces the original C1 claim ("MLL-23 Normal-recall 0.998") which was
%% joint-training in-distribution and therefore not a cross-cohort result.
\subsection{Multi-cohort iVAE training under strict held-out evaluation (C1$'$)}
\label{sec:c1_loco}

\paragraph{Why this replaces the original C1 block.}
The original C1 protocol trained jointly on \mbox{(GR-Neutro\,+\,MLL-23)}
with a cohort-ID auxiliary head and reported a Normal-recall of $0.998$ on
MLL-23. That number was \emph{joint-training in-distribution}: MLL-23 cells
were part of the training mix, so it cannot be compared against the
published pure-CBM ($0.473$) or Concept-Distribution-Alignment ($0.550$)
figures, which are strict zero-shot transfers from GR-Neutro to a
never-seen MLL-23 cohort. We retract the cross-cohort framing of the
``$0.998$'' number and replace it with the leave-one-cohort-out (LOCO)
evaluation below.

\paragraph{LOCO protocol.}
For each held-out cohort $u_h\in\{$GR-Neutro,\,AML-Matek,\,MLL-23,\,Bodzas$\}$,
training pools the other three cohorts; the auxiliary head learns only the
three training-cohort identities. The held-out cohort never enters the
training mix --- enforced at the data-loader level and asserted in
\texttt{train\_loco.py}. The held-out cohort's class-head is COLD at test
time, so Normal-recall is computed by routing concept-prob outputs through
the GR-Neutro Normal prototype row (the same fair-comparison rule as the
published pure-CBM / CDA baselines). Each config is trained with $3$ seeds.

\begin{table}[h]
\centering
\small
\caption{LOCO held-out Normal-recall, in-distribution pooled W-F1 / Macro-F1
on the three training cohorts, and concept-singular-value decompression on
the held-out cohort (number of singular values above $0.05$ of the leading
value, and the $\sigma_7-\sigma_8$ gap). Three-seed mean\,$\pm$\,sd.}
\label{tab:c1_loco}
\begin{tabular}{llccccc}
\toprule
Config & Held-out & Normal-recall $\uparrow$ & W-F1 $\uparrow$ & Macro-F1 $\uparrow$ & $n_{\sigma\ge 0.05}$ & $\sigma_7-\sigma_8$ \\
\midrule
__ROWS__
\bottomrule
\end{tabular}
\end{table}

\paragraph{Honest verdict --- Config~C is the canonical comparator.}
Config~C (hold out MLL-23) is the apples-to-apples comparator for the
published pure-CBM Normal-recall on MLL-23 ($0.473$) and CDA Normal-recall
on MLL-23 ($0.550$). __VERDICT_C__

Config~D (hold out Bodzas) probes the most stringent transfer: the model
never saw Bodzas cells nor its staining distribution, but did see the
most-similar peripheral-blood smear cohort (GR-Neutro) plus AML~Matek and
MLL-23 during training. __VERDICT_D__

The concept rank-decompression columns ($n_{\sigma\ge 0.05}$ and the
$\sigma_7-\sigma_8$ gap) test the identifiability claim of Khemakhem~et~al.
(2020): under sufficient cohort variation in the auxiliary $u$ the per-cell
concept axes should decompress beyond the $K=7$ class-collapsed rank
described by Lemma~1 / Corollary~1. We report these values on the held-out
cohort to keep them off the supervision path.

\paragraph{In-distribution trade-off (honest disclosure).}
The original (joint) C1 reported an in-distribution loss of
$-2.8$\,pp W-F1 and $-7.4$\,pp Macro-F1 on GR-Neutro relative to the pure
single-cohort CBM ($0.887\to0.859$ W-F1; $0.842\to0.768$ Macro-F1). The LOCO
in-distribution columns in Table~\ref{tab:c1_loco} are reported on the
\emph{three training cohorts pooled} (the held-out cohort is, by
construction, excluded from these metrics).

\paragraph{What the original ``$0.998$'' headline really was.}
A multi-task supervised artefact: the cohort-ID auxiliary head lets the
model freely shape its concept space around MLL-23 cells that were
literally in the training batch. Removing MLL-23 from training removes
this lever, so the strict-LOCO MLL-23 Normal-recall (Config~C) is the
only number that can be honestly compared against the published
$0.473$ / $0.550$ baselines.
"""
    tex = tex.replace("__ROWS__", rows)

    # Verdict text from Config C.
    def verdict_c():
        if "C" not in cfgs or cfgs["C"]["aggregate"]["held_normal_recall"] is None:
            return "Config~C has not yet completed."
        m = cfgs["C"]["aggregate"]["held_normal_recall"]["mean"]
        if m > pub["mll23_cda_normal_recall"]:
            return (f"Under LOCO, Config~C reaches Normal-recall ${m:.3f}$ on held-out "
                    f"MLL-23, exceeding both the published pure-CBM baseline ($0.473$) "
                    f"and the CDA baseline ($0.550$). The constructive iVAE claim "
                    f"holds in the strict zero-shot sense.")
        elif m > pub["mll23_pure_cbm_normal_recall"]:
            return (f"Under LOCO, Config~C reaches Normal-recall ${m:.3f}$ on held-out "
                    f"MLL-23, above the pure-CBM baseline ($0.473$) but below the CDA "
                    f"baseline ($0.550$). The iVAE conditioning helps but does not "
                    f"surpass distribution-alignment-based transfer.")
        else:
            return (f"Under LOCO, Config~C reaches Normal-recall ${m:.3f}$ on held-out "
                    f"MLL-23 --- at or below the published pure-CBM baseline ($0.473$). "
                    f"The original C1 ``0.998'' headline was a joint-training artefact: "
                    f"removing MLL-23 from the training mix removes the gain. The "
                    f"constructive iVAE claim does not survive LOCO and we retract it.")

    def verdict_d():
        if "D" not in cfgs or cfgs["D"]["aggregate"]["held_normal_recall"] is None:
            return "Config~D has not yet completed."
        m = cfgs["D"]["aggregate"]["held_normal_recall"]["mean"]
        return (f"Config~D reaches Normal-recall ${m:.3f}$ on held-out Bodzas, the "
                f"most stringent transfer (the model has not seen Bodzas cells nor "
                f"the Bodzas staining distribution).")

    tex = tex.replace("__VERDICT_C__", verdict_c())
    tex = tex.replace("__VERDICT_D__", verdict_d())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
