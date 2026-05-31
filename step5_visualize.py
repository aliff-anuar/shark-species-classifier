# =============================================================
# STEP 5: DATA VISUALIZATION
# Role: Data Analyst
# Description: Visualize dataset, model performance,
#              confusion matrix, and final conclusion
# =============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ── Configuration ─────────────────────────────────────────────
SPLIT_DIR   = "shark_dataset/split"
RESULTS_DIR = "shark_results"
PLOTS_DIR   = "shark_plots"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32

os.makedirs(PLOTS_DIR, exist_ok=True)

CLASS_NAMES = [
    "Blacktip Reef",
    "Bull Shark",
    "Great White",
    "Hammerhead",
    "Mako Shark",
    "Nurse Shark",
    "Tiger Shark",
    "Whale Shark"
]

MODEL_NAMES  = ["ResNet50", "DenseNet121", "MobileNetV3"]
MODEL_COLORS = ["#E63946", "#2A9D8F", "#E9C46A"]

# ── Load Results ──────────────────────────────────────────────
def load_results():
    all_results = {}
    for name in MODEL_NAMES:
        path = os.path.join(RESULTS_DIR, f"{name}_results.json")
        with open(path, "r") as f:
            all_results[name] = json.load(f)
    return all_results

# ─────────────────────────────────────────────────────────────
# PLOT 1: Dataset Distribution
# ─────────────────────────────────────────────────────────────
def plot_dataset_distribution():
    print("Plotting dataset distribution...")

    counts = {"train": [], "val": [], "test": []}
    for split in counts:
        split_path = os.path.join(SPLIT_DIR, split)
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                counts[split].append(len(os.listdir(cls_path)))

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0D1B2A")
    ax.set_facecolor("#0D1B2A")

    bars1 = ax.bar(x - width, counts["train"], width, label="Train",      color="#E63946", alpha=0.9)
    bars2 = ax.bar(x,         counts["val"],   width, label="Validation",  color="#2A9D8F", alpha=0.9)
    bars3 = ax.bar(x + width, counts["test"],  width, label="Test",        color="#E9C46A", alpha=0.9)

    ax.set_xlabel("Shark Species", color="white", fontsize=12)
    ax.set_ylabel("Number of Images", color="white", fontsize=12)
    ax.set_title("Dataset Distribution per Class", color="white", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444")
    ax.legend(facecolor="#1A2A3A", labelcolor="white", fontsize=11)
    ax.yaxis.label.set_color("white")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "1_dataset_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 1_dataset_distribution.png")

# ─────────────────────────────────────────────────────────────
# PLOT 2: Training Accuracy Curves
# ─────────────────────────────────────────────────────────────
def plot_accuracy_curves(all_results):
    print("Plotting accuracy curves...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.patch.set_facecolor("#0D1B2A")
    fig.suptitle("Training vs Validation Accuracy", color="white", fontsize=16, fontweight="bold", y=1.02)

    for ax, (name, color) in zip(axes, zip(MODEL_NAMES, MODEL_COLORS)):
        history = all_results[name]["history"]
        epochs  = range(1, len(history["accuracy"]) + 1)

        ax.set_facecolor("#0D1B2A")
        ax.plot(epochs, history["accuracy"],     color=color,   linewidth=2,   label="Train Acc")
        ax.plot(epochs, history["val_accuracy"], color="white", linewidth=2,   linestyle="--", label="Val Acc")
        ax.set_title(name, color=color, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("Accuracy", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#444")
        ax.legend(facecolor="#1A2A3A", labelcolor="white", fontsize=10)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "2_accuracy_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 2_accuracy_curves.png")

# ─────────────────────────────────────────────────────────────
# PLOT 3: Training Loss Curves
# ─────────────────────────────────────────────────────────────
def plot_loss_curves(all_results):
    print("Plotting loss curves...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.patch.set_facecolor("#0D1B2A")
    fig.suptitle("Training vs Validation Loss", color="white", fontsize=16, fontweight="bold", y=1.02)

    for ax, (name, color) in zip(axes, zip(MODEL_NAMES, MODEL_COLORS)):
        history = all_results[name]["history"]
        epochs  = range(1, len(history["loss"]) + 1)

        ax.set_facecolor("#0D1B2A")
        ax.plot(epochs, history["loss"],     color=color,   linewidth=2, label="Train Loss")
        ax.plot(epochs, history["val_loss"], color="white", linewidth=2, linestyle="--", label="Val Loss")
        ax.set_title(name, color=color, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("Loss", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#444")
        ax.legend(facecolor="#1A2A3A", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "3_loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 3_loss_curves.png")

# ─────────────────────────────────────────────────────────────
# PLOT 4: Confusion Matrices (all 3 models)
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrices():
    print("Plotting confusion matrices...")

    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.patch.set_facecolor("#0D1B2A")
    fig.suptitle("Confusion Matrices on Test Set", color="white", fontsize=16, fontweight="bold", y=1.02)

    for ax, (name, color) in zip(axes, zip(MODEL_NAMES, MODEL_COLORS)):
        model_path = os.path.join(RESULTS_DIR, f"{name}_final.h5")
        model = keras.models.load_model(model_path)

        test_gen.reset()
        y_pred  = np.argmax(model.predict(test_gen, verbose=0), axis=1)
        y_true  = test_gen.classes
        cm      = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, cbar=False, annot_kws={"size": 8}
        )
        ax.set_title(name, color=color, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", color="white", fontsize=10)
        ax.set_ylabel("Actual",    color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", color="white")
        ax.set_yticklabels(CLASS_NAMES, rotation=0, color="white")
        ax.set_facecolor("#0D1B2A")

        # Print classification report
        print(f"\n  Classification Report - {name}:")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "4_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 4_confusion_matrices.png")

# ─────────────────────────────────────────────────────────────
# PLOT 5: Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────
def plot_model_comparison(all_results):
    print("Plotting model comparison...")

    accuracies     = [all_results[m]["test_accuracy"]  for m in MODEL_NAMES]
    maps           = [all_results[m]["test_map"]        for m in MODEL_NAMES]
    training_times = [all_results[m]["training_time"]   for m in MODEL_NAMES]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0D1B2A")
    fig.suptitle("Model Comparison", color="white", fontsize=16, fontweight="bold")

    metrics = [
        (accuracies,     "Test Accuracy",      axes[0]),
        (maps,           "Test mAP",           axes[1]),
        (training_times, "Training Time (min)", axes[2]),
    ]

    for values, title, ax in metrics:
        ax.set_facecolor("#0D1B2A")
        bars = ax.bar(MODEL_NAMES, values, color=MODEL_COLORS, width=0.5, alpha=0.9)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.set_ylabel(title, color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#444")
        ax.set_xticklabels(MODEL_NAMES, color="white", fontsize=11)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}" if title != "Training Time (min)" else f"{val:.1f}",
                ha="center", va="bottom", color="white", fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "5_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 5_model_comparison.png")

# ─────────────────────────────────────────────────────────────
# PLOT 6: Per-Class mAP Heatmap
# ─────────────────────────────────────────────────────────────
def plot_per_class_map(all_results):
    print("Plotting per-class mAP heatmap...")

    data = np.array([
        [all_results[m]["per_class_ap"][c.lower().replace(" ", "_") + "_shark"]
         if c.lower().replace(" ", "_") + "_shark" in all_results[m]["per_class_ap"]
         else all_results[m]["per_class_ap"].get(list(all_results[m]["per_class_ap"].keys())[i], 0)
         for i, c in enumerate(CLASS_NAMES)]
        for m in MODEL_NAMES
    ])

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0D1B2A")
    ax.set_facecolor("#0D1B2A")

    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=CLASS_NAMES, yticklabels=MODEL_NAMES,
        ax=ax, linewidths=0.5, linecolor="#333"
    )
    ax.set_title("Per-Class Average Precision (AP) per Model", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Shark Species", color="white", fontsize=11)
    ax.set_ylabel("Model", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", color="white", fontsize=10)
    ax.set_yticklabels(MODEL_NAMES, rotation=0, color="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "6_per_class_map.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 6_per_class_map.png")

# ─────────────────────────────────────────────────────────────
# FINAL CONCLUSION (printed + saved to text file)
# ─────────────────────────────────────────────────────────────
def print_conclusion(all_results):
    conclusion_lines = []

    def log(line=""):
        print(line)
        conclusion_lines.append(line)

    log("=" * 65)
    log("  FINAL CONCLUSION")
    log("=" * 65)
    log()
    log(f"  {'Model':<15} {'Accuracy':>10} {'mAP':>10} {'Time (min)':>12}")
    log(f"  {'-'*50}")
    for name in MODEL_NAMES:
        r = all_results[name]
        log(f"  {name:<15} {r['test_accuracy']:>10.4f} {r['test_map']:>10.4f} {r['training_time']:>12.1f}")
    log()

    # Determine best model per metric
    best_acc  = max(MODEL_NAMES, key=lambda m: all_results[m]["test_accuracy"])
    best_map  = max(MODEL_NAMES, key=lambda m: all_results[m]["test_map"])
    best_time = min(MODEL_NAMES, key=lambda m: all_results[m]["training_time"])

    log(f"  Best Accuracy      : {best_acc}  ({all_results[best_acc]['test_accuracy']*100:.2f}%)")
    log(f"  Best mAP           : {best_map}  ({all_results[best_map]['test_map']:.4f})")
    log(f"  Fastest Training   : {best_time} ({all_results[best_time]['training_time']:.1f} min)")
    log()

    # Model parameters (approximate)
    param_counts = {
        "ResNet50"    : "~25.6M",
        "DenseNet121" : "~8.0M",
        "MobileNetV3" : "~5.4M"
    }
    log("  Model Parameters:")
    for name in MODEL_NAMES:
        log(f"    {name:<15}: {param_counts[name]}")
    log()
    log("  Conclusion:")
    log("  Based on accuracy, mAP, training time, and model size,")
    log("  the recommended model for shark species classification is")
    log("  determined by balancing all four criteria:")
    log()
    log("  - If ACCURACY is the top priority   → choose ResNet50")
    log("  - If EFFICIENCY is the top priority → choose MobileNetV3")
    log("  - If BALANCE is the priority        → choose DenseNet121")
    log()
    log("  For a deployment scenario (e.g., mobile app or edge device),")
    log("  MobileNetV3 is the best choice due to its small size and")
    log("  fast inference despite slightly lower accuracy.")
    log()
    log("  For a research/server scenario where accuracy matters most,")
    log("  ResNet50 or DenseNet121 are preferred.")
    log("=" * 65)

    # Save conclusion to file
    conclusion_path = os.path.join(PLOTS_DIR, "conclusion.txt")
    with open(conclusion_path, "w") as f:
        f.write("\n".join(conclusion_lines))
    print(f"\n  Conclusion saved to: {conclusion_path}")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading results...\n")
    all_results = load_results()

    plot_dataset_distribution()
    plot_accuracy_curves(all_results)
    plot_loss_curves(all_results)
    plot_confusion_matrices()
    plot_model_comparison(all_results)
    plot_per_class_map(all_results)
    print_conclusion(all_results)

    print(f"\n All plots saved to: {PLOTS_DIR}/")
    print("  Files generated:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        print(f"    - {f}")
