import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix
cm = np.array([
    [1279,  88, 151, 118],
    [  57, 765, 179,  83],
    [ 136, 204,1229, 139],
    [  78,  30,  72, 923]
])

labels = ["Happy", "Sad", "Neutral", "Angry"]

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 11}
)

plt.xlabel("Predicted Emotion", fontsize=11)
plt.ylabel("True Emotion", fontsize=11)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.savefig("confusion_matrix.pdf", dpi=300, bbox_inches="tight")
plt.close()