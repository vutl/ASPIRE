**Response to Reviewer gRZQ**

We sincerely thank the reviewer for the constructive feedback and for recognizing the clarity of our motivation, the completeness of the framework, and the strength of our extensive evaluations. Below, we address your concerns point by point.

**(1) Scope of the contribution and technical novelty**
We acknowledge your point that ASPIRE utilizes established base modules (e.g., WavLM, RoBERTa). However, we respectfully emphasize that our core technical novelty lies not in the individual encoders, but in the **hierarchical multi-task formulation** and the **geometric guidance mechanism**. Rather than simply concatenating generic features, ASPIRE explicitly aligns disentangled prosody (arousal-centric) and semantics (valence-centric) through bidirectional cross-attention to estimate continuous affective coordinates. These coordinates are then transformed into a spatial guidance vector ($e_{AV}$) that actively reshapes the final categorical decision boundary. We will revise the wording to make the contribution clearer.

**(2) Fairness of Table 1**
We completely agree that Table 1 can be misleading when read as the primary model comparison, given that several baseline entries are audio-only while ASPIRE is multimodal. Our initial intent was to use Table 1 to establish the baseline performance of strong SSL acoustic encoders before introducing A/V guidance and semantic cues. To ensure a fair and transparent evaluation protocol, we will restructure the results in the revision. Multimodal audio-text comparisons (currently in Table 2) will be promoted to the main comparison table, while audio-only SSL models will be clearly demarcated as the initial acoustic baseline group.

**(3) Final fusion strategy**
We appreciate you pointing out the need for a deeper discussion on the final fusion choice. To address this, we conducted a controlled ablation on IEMOCAP where the encoders, cross-attention module, and A/V refinement module are frozen; only the final fusion operator combining the representations ($e_{Global}$, $e_A$, $e_V$, and $e_{AV}$) is changed.

| Fusion Strategy | Acc. | Macro-F1 | Arousal CCC | Valence CCC |
| --- | --- | --- | --- | --- |
| Original concat | 0.7586 | **0.7631** | 0.7140 | 0.7400 |
| Sum fusion | **0.7592** | 0.7462 | **0.7264** | 0.7313 |
| Gated fusion | 0.7232 | 0.7135 | 0.6785 | **0.7453** |
| Attention fusion | 0.7057 | 0.6930 | 0.6813 | 0.7228 |

The results provide a clear justification: sum fusion yields competitive accuracy but lower F1, while gated and attention mechanisms perform noticeably worse for categorical classification. Our interpretation is that simple concatenation preserves the distinct, specialized properties of the four branches ($e_{Global}$, $e_A$, $e_V$, $e_{AV}$), making them fully accessible to the final classifier. Conversely, gated or attention fusions compress these distinct signals too early, which slightly aids a single regression dimension (e.g., Valence CCC in Gated fusion) but harms overall discrete emotion classification. We do not claim concatenation is universally superior, but it is the most stable choice for ASPIRE's multi-stream architecture. We will add this ablation and expand our discussion on the fusion design.

**(4) Clarification of Figure 4 (UMAP)**
We apologize for the ambiguity in Figure 4. The UMAP plot visualizes the **final fused representation ($e_{final}$)** immediately prior to the categorical classifier (i.e., the concatenation of the global acoustic branch, arousal branch, valence branch, and A/V-guided vector). It does not represent a single raw modality like WavLM or RoBERTa. For reproducibility, we standardized the features across all five IEMOCAP folds before projection and utilized the standard UMAP hyperparameters: `random_state=42`, `n_neighbors=15` (to capture local manifold structure), and `min_dist=0.3` (to provide a moderate spread without forcing overly tight clusters). We will explicitly state the representation source and the full UMAP protocol in the revised caption.

**(5) Presentation issues**
We thank the reviewer for their meticulous reading. We will thoroughly proofread the manuscript and correct the minor grammar and spelling errors in the final revision.