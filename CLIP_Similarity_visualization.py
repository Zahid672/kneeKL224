import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_text_sim_heatmap(sim_matrix, labels, title="Text–Image Similarity"):
    plt.figure(figsize=(8,6))
    sns.heatmap(sim_matrix, annot=True, cmap="viridis",
                xticklabels=labels, yticklabels=labels,
                fmt=".2f", cbar=True)
    plt.xlabel("Text Prompt")
    plt.ylabel("Image Prediction")
    plt.title(title)
    plt.tight_layout()
    plt.show()
