import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_history(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_comparison(histories, labels, metric, out_path, title):
    plt.figure()
    for h, label in zip(histories, labels):
        plt.plot(h[metric], label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()


def compare_all(outputs_dir="outputs"):
    outputs_dir = Path(outputs_dir)
    modes = ["frozen", "partial", "full"]

    histories = [
        load_history(outputs_dir / mode / "history.json")
        for mode in modes
    ]

    plot_comparison(
        histories,
        modes,
        metric="val_acc",
        out_path=outputs_dir / "compare_val_accuracy.png",
        title="Validation Accuracy Comparison"
    )

    plot_comparison(
        histories,
        modes,
        metric="train_loss",
        out_path=outputs_dir / "compare_train_loss.png",
        title="Training Loss Comparison"
    )
