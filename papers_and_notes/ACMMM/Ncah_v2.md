**Response to Reviewer Ncah**

We sincerely thank the reviewer for evaluating our work, recognizing that it aligns well with ACM Multimedia's core topics, and acknowledging its performance on IEMOCAP and MSP-Podcast. Below, we address your concerns in detail.

**1. Clarification on Model Contributions and Novelty (Addressing Weaknesses 1 & 2)**
We appreciate the reviewer’s perspective. While it is true that we utilize established encoders (W-Pro, RoBERTa) and cross-attention, **our core contribution does not lie in the individual modules, but in the novel architectural routing and the Geometric Guidance mechanism.**

Prior works typically concatenate acoustic and textual features into a generic representation and treat dimensional prediction (Arousal-Valence) as a loosely coupled auxiliary loss. ASPIRE departs from this standard multi-task paradigm in two fundamental ways:

* **Functional Asymmetry:** Grounded in psychoacoustic evidence, we explicitly force the model to route prosody to drive Arousal and semantics to inform Valence, avoiding the entanglement of standard fusion methods.
* **Geometric Inductive Bias:** The true innovation is our **Dimensional Guidance Module (DGM)**. Instead of leaving categorical prediction and dimensional prediction completely separated, we transform the predicted continuous AV coordinates into an internal *geometric guidance vector* ($e_{AV}$). This vector actively reshapes and refines the discrete categorical decision boundaries. As demonstrated in our Ablation Study (Table 4), removing this DGM structure causes a severe performance drop (from 75.86% to 71.40% Accuracy), proving that the cooperative geometric refinement—not the simple combination of backbones—is the primary driver of our model's success.

**2. Fairness of Comparison in Table 1 (Addressing Weakness 4)**
We fully agree with the reviewer that the current presentation in Table 1 is misleading. Table 1 was initially designed to demonstrate how our proposed framework builds upon and exceeds strong acoustic self-supervised (SSL) baselines before introducing semantic cues.

However, we acknowledge that comparing a multimodal (A+T) system against unimodal baselines in the main results table creates an unfair impression. In the revision, we will comprehensively restructure the results section. We will move the multimodal audio-text comparisons (currently in Table 2) to Table 1 as the primary benchmark, and group the audio-only systems into a separate "Acoustic-baseline" block. This will ensure absolute clarity and fairness in our evaluation protocol.

**3. Comparisons with Recent Literature (Addressing Weakness 5)**
We thank the reviewer for the suggestion to include more recent works. First, we would like to respectfully point out that our original manuscript (Table 2) **already includes several recent multimodal methods published within the last two years (2024–2026)** under a comparable IEMOCAP 4-class setting, specifically:

* **AFEA-Net (2025)** [30]
* **MSF-SER (2025)** [17]
* **PCM (2025)** [35]
* **WavLM-LR (2024)** [24]
* **CRAB (2026)** [42]

To further strengthen this evaluation and provide a broader context, we have supplemented our comparisons with additional recent multimodal SER methods (e.g., FocalSer, MemoCMT, Medusa). The numbers below are sourced from the recent evaluation protocol in CRAB [42], which is a highly relevant reference as it also utilizes a bimodal WavLM + RoBERTa cross-modal setup on both IEMOCAP and MSP-Podcast v2.0.

| Model | IEMOCAP 4-cls WAR | MSP-Test1 8-cls WAR | MSP-Test2 8-cls WAR |
| --- | --- | --- | --- |
| FocalSer (2024) | 0.6676 | 0.2892 | 0.1981 |
| MemoCMT (2025) | 0.6964 | 0.6239 | 0.6058 |
| Medusa (2025) | 0.6605 | 0.4157 | 0.4226 |
| CRAB (2026) | 0.7348 | 0.5232 | 0.4861 |
| **ASPIRE (Ours)** | **0.7586** | **0.6445** | **0.6430** |

*Note: Since recent SER papers often differ in label taxonomy, dataset splits, or transcript sources, exact replications under a single unified protocol are challenging. In our revision, we will clearly categorize comparisons into (i) directly comparable same-protocol results and (ii) contextual recent-method comparisons.*

**4. Contextualizing LLM-based and AudioLLM-based SER (Addressing Weakness 3)**
We agree that acknowledging the rapid progress in audio-language large models and LLM-based SER is crucial for a complete literature review. We originally omitted them from the main comparison tables because they operate under fundamentally different paradigms and evaluation protocols compared to ASPIRE (e.g., zero-shot omni-LLM inference, transcript-based prompting, different label sets, and LLM-as-a-judge scoring on a 0-100 scale rather than standard F1). Directly ranking our 6.35M-parameter model alongside billion-parameter AudioLLMs in the same table would be mathematically and structurally misleading.

Nevertheless, following your constructive advice, we will add a dedicated discussion and a separate contextual table to explicitly position our method against this emerging LLM-based paradigm:

| Group | Method | Input / Protocol | IEMOCAP Result |
| --- | --- | --- | --- |
| **Ours** | **ASPIRE** | **A+T, 4-class 5-fold SER** | **75.85 W-F1** |
| LLM-based SER | SpeechCueLLM (2025) | LLaMA-2 + transcript + speech descriptions | 72.60 W-F1 |
| LLM + speech encoder | LLM + FF Projector (2025) | Data2Vec + LLaMA-2 LoRA | 70.64 W-F1 |
| Zero-shot omni-LLM | OmniVox / GPT-4o-Audio (2025) | raw audio, 6-class, context c=12 | 55.90 W-F1 |
| Zero-shot omni-LLM | OmniVox / Gemini-Lite (2025) | raw audio, 6-class, context c=12 | 54.10 W-F1 |
| AudioLLM | WavLLM (2025) | AudioBench ER, LLM-as-a-judge | 59.80 score* |
| AudioLLM | Qwen2-Audio (2025) | AudioBench ER, LLM-as-a-judge | 54.00 score* |

*(Note: "score*" denotes the AudioBench-style evaluation for open-ended reasoning rather than standard classification accuracy/F1).*

We hope this response clarifies our contributions and demonstrates our commitment to presenting a fair and rigorous comparison. We sincerely hope you will consider raising your score based on these clarifications.

**References for LLM-based paradigm:**
Z. Wu et al., "Beyond Silent Letters: Amplifying LLMs in Emotion Recognition with Vocal Nuances," 2025.  
J. Murzaku and O. Rambow, "OmniVox: Zero-Shot Emotion Recognition with Omni-LLMs," 2025.  
W. Zhang et al., "Beyond Classification: Towards Speech Emotion Reasoning with Multitask AudioLLMs," 2025.  
B. Wang et al., "AudioBench: A Universal Benchmark for Audio Large Language Models," NAACL, 2025.