**Response to Reviewer hxpS**

We sincerely thank the reviewer for the precise and highly constructive comments. We deeply appreciate your recognition of our explicit modeling of feature asymmetry and the geometrically motivated design. Below, we address your concerns regarding evaluation granularity, the oracle baseline, and computational cost.

**(1) Fine-grained emotion boundaries**
We agree that the standard 4-class IEMOCAP setting does not fully stress-test boundary refinement. Our original 4-class setup was chosen for fair comparison, as prior work recommends merging *Happy* and *Excited* due to their proximity in the activation-valence space [1, 2, 3]. To directly address your concern, we additionally evaluated ASPIRE on harder 5-class and 6-class IEMOCAP settings by separating *Happy*/*Excited* and adding *Frustrated*. We compare against MemoCMT [4], a recent audio-text emotion recognition baseline evaluated on IEMOCAP.

| Model | Setting | Acc. | Macro-F1 |
| --- | --- | --- | --- |
| MemoCMT [4] | 5-class | 0.6644 | 0.6227 |
| **ASPIRE-AV** | **5-class** | **0.6934** | **0.6651** |
| MemoCMT [4] | 6-class | 0.5553 | 0.5351 |
| **ASPIRE-AV** | **6-class** | **0.5908** | **0.5838** |
| **ASPIRE-AVD** | **6-class** | **0.5985** | **0.5913** |

These results show that ASPIRE remains robust under more crowded affective labels. Inspecting the confusion matrices further clarifies our claim:

* **5-class ASPIRE-AV**: `hap->hap=228/595`, `hap->exc=140`, `exc->exc=624/1041`, `exc->hap=95`.

* **6-class ASPIRE-AV**: `ang->ang=776/1103`, `ang->fru=223`, `fru->fru=1102/1849`, `fru->ang=321`, `fru->neu=264`.

ASPIRE does not completely eliminate confusion between naturally adjacent emotions, as *Happy* and *Excited* inherently overlap in positive valence. Similarly, *Frustrated* remains confused with *Angry*, *Neutral*, and *Sad* due to shared appraisal characteristics [6]. Our claim is that AV/AVD guidance provides additional geometric cues that meaningfully improve decision boundaries in these overlapping regions, not that it perfectly separates them.

**(2) Ground-truth A/V upper-bound**
We agree that an oracle diagnostic is needed to reveal the theoretical ceiling of the guidance mechanism. We compared: (i) no dimensional guidance, (ii) predicted guidance, and (iii) ground-truth guidance. The oracle diagnostic replaces $[\hat v_A,\hat v_V]$ in our refinement equation with ground-truth coordinates:

$$e_{AV}=MLP_{\text{ref}}([\hat v_A,\hat v_V]) + W_{\text{res}}[\hat v_A,\hat v_V]$$

| Dataset | Guid. | No guid. | Pred. guid. | GT upper |
| --- | --- | --- | --- | --- |
| IEMOCAP 4-cls | AV | 0.7303 | 0.7586 | **0.7803** |
| IEMOCAP 6-cls | AVD | 0.5744 | 0.5985 | **0.6058** |
| MSP Test1 8-cls | AV | 0.6242 | 0.6445 | **0.6595** |

Predicted guidance consistently improves over no guidance, while GT guidance further establishes the upper bound. This confirms that the geometric guidance mechanism operates effectively as theorized, and demonstrates additional headroom as A/V prediction accuracy improves.

**(3) Computational Cost and Inference Efficiency**
Regarding the efficiency claim, while ASPIRE utilizes three pre-trained backbones, we have transparently reported the high computational cost (>200 Giga-FLOPS) in Table 4 of our manuscript. Our claim of efficiency specifically targets the *trainable parameters* (6.35M) during the adaptation phase. Freezing the massive backbones drastically reduces training memory requirements. During inference, while the total FLOPs are high, the parallel nature of our three independent branches (W-Pro, RoBERTa, WavLM) allows them to be executed concurrently, effectively mitigating latency bottlenecks in real-world deployments. We will explicitly clarify this trade-off between trainable parameter efficiency and inference FLOPs in the revised manuscript.

**References:**
[1] C. Busso et al., "IEMOCAP: Interactive Emotional Dyadic Motion Capture Database," *Language Resources and Evaluation*, 2008.
[2] E. Lakomkin et al., "On the Robustness of Speech Emotion Recognition for Human-Robot Interaction with Deep Neural Networks," *IROS*, 2018.
[3] N. Antoniou et al., "Designing and Evaluating Speech Emotion Recognition Systems: A Reality Check with IEMOCAP," *ICASSP*, 2023.
[4] M. Khan et al., "MemoCMT: Multimodal Emotion Recognition Using Cross-Modal Transformer-Based Feature Fusion," *Scientific Reports*, 2025.
[6] P. Kuppens et al., "The Appraisal Basis of Anger: Specificity, Necessity, and Sufficiency of Components," *Emotion*, 2003.