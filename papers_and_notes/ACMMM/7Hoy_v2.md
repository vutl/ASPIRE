### Rebuttal to Reviewer 7Hoy

We sincerely thank Reviewer 7Hoy for the careful and constructive assessment. We are highly encouraged that you recognize ASPIRE's psychoacoustic grounding, our comprehensive ablation design, and the model's strong parameter efficiency. Below, we address your valuable feedback point-by-point:

**1. Unfair Comparison Structure and Multimodal Baselines on MSP-Podcast**
We completely agree with your critique regarding Table 1. Comparing our multimodal framework against unimodal (audio-only) SSL baselines in the primary table creates a structural mismatch. Our original intention was to demonstrate the acoustic starting point, but we acknowledge this makes the comparison appear unfair.

* **Restructuring:** In the revised manuscript, we will merge the audio-only SSL models into a separate "Acoustic Baseline" section within the multimodal comparison table. We will elevate the same-scope A+T (multimodal) comparison to be the primary result table and clarify that missing entries for prior works are due to metrics not being reported by the original authors.
* **Direct Multimodal Comparison on MSP-Podcast:** We agree that a direct multimodal comparison on MSP-Podcast is crucial. Since many IEMOCAP A+T systems (e.g., MTAF, AFEA-Net) do not report on MSP-Podcast v2.0 or release adaptable code, we have conducted new comparisons against recent multimodal baselines reported in CRAB [1]—one of the few works evaluating on both datasets. As shown below, we report the 8-class WAR (Weighted Accuracy, equivalent to standard accuracy) to align with prior work protocols:

| Model | MSP-Test1 (8-cls WAR) | MSP-Test2 (8-cls WAR) |
| --- | --- | --- |
| FocalSER [1] | 0.2892 | 0.1981 |
| Medusa [1] | 0.4157 | 0.4226 |
| CRAB [1] | 0.5232 | 0.4861 |
| MemoCMT [1] | 0.6239 | 0.6058 |
| **ASPIRE (Ours)** | **0.6445** | **0.6430** |

This direct comparison clearly demonstrates ASPIRE's superiority over existing multimodal approaches on in-the-wild datasets. We will include this table in the revision, noting that exact preprocessing may still slightly vary across papers.

**2. Discrepancy Between MSP Test1 and Test2 Performance**
You raised an insightful question regarding the macro-F1 drop between Test1 and Test2 (67.58 vs. 52.93). We will explicitly discuss this in the revision. MSP-Podcast v2.0 is a highly naturalistic corpus. Crucially, **Test2 is constructed differently from Test1**: it was collected *without* the emotion-retrieval step, leading to a much higher proportion of Neutral samples (~45.8%) and notably lower inter-evaluator agreement [2].
Because Macro-F1 is highly sensitive to class imbalance, this drop reflects the severe imbalance and annotation variability in Test2 rather than a failure in our model's generalization. This is empirically supported by our new 8-class WAR results (Table above), which remain highly stable across Test1 (0.6445) and Test2 (0.6430).

**3. Rhetorical Framing around $e_{AV}$**
We appreciate this candid feedback and agree that terms like "affective compass" and "geometric guidance" may sound overly rhetorical. As your insightful reading of our ablations confirms, the primary benefit derives from structured multi-task learning, branch specialization, and cross-modal interaction, rather than just the isolated $e_{AV}$ step. We will tone down the rhetoric and revise the manuscript to describe $e_{AV}$ more accurately as an **"AV-conditioned refinement vector"**, ensuring the text honestly reflects the mechanistic contributions shown in the ablation studies.

**4. W-Pro Dependency and Generalizability**
We agree that the dependency on W-Pro requires critical reflection. What W-Pro captures that standard WavLM Base+ does not is the *explicit suppression of global timbre and speaker identity*, leaving a "pure" prosodic signal (pitch, rhythm, energy) that prevents the model from overfitting to speaker-specific artifacts.
While advanced prosody disentanglement methods exist (e.g., Prosody2Vec [3], Diffsody [4]), they are designed for reconstruction or voice conversion and are not readily available as drop-in extractors for SER. However, to prove that our framework's success is **not strictly dependent on W-Pro**, we refer to **Table 5** in our manuscript. Replacing W-Pro with the lightweight OpenSMILE (eGeMAPS) still yields a highly competitive accuracy of 75.17%, while generic WavLM achieves 74.78%. ASPIRE does not theoretically require W-Pro; rather, W-Pro is currently the strongest available drop-in prosody encoder that maximizes the effectiveness of our semantic-prosody interaction. We will expand this reflection in the revised text.

We hope these clarifications and new empirical baseline comparisons thoroughly address your concerns.

**References:**
[1] L. H. Ueda et al., "CRAB: Multi Layer Contrastive Supervision to Improve Speech Emotion Recognition Under Both Acted and Natural Speech Condition," 2026.  
[2] C. Busso et al., "The MSP-Podcast Corpus", 2025.
[3] L. Qu et al., "Disentangling Prosody Representations with Unsupervised Speech Reconstruction," 2022.
[4] L. Qu et al., "Disentanglement of Prosody Representations via Diffusion Models and Scheduled Gradient Reversal," 2025.