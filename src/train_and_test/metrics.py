import json
import os

import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

def plot_composite_metric(epochs, history, output_dir, target_name, colors):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Ploting the components_output
    axes[0].plot(epochs, history[f"{target_name}_output_mcc_manual"], label="Train Components", color=colors[0])
    axes[0].plot(epochs, history[f"val_{target_name}_output_mcc_manual"], label="Val Components", color=colors[0], linestyle="dashed")
    axes[0].set_title(f"MMC - {target_name.capitalize()} Output")
    axes[0].set_ylabel('MMC')
    axes[0].legend()
    axes[0].grid()
    
    # Ploting the type_output and cause_output
    axes[1].plot(epochs, history[f"{target_name}_output_pr_auc"], label="Train Type", color=colors[1])
    axes[1].plot(epochs, history[f"val_{target_name}_output_pr_auc"], label="Val Type", color=colors[1], linestyle="dashed")
    
    axes[1].plot(epochs, history[f"{target_name}_output_f1_score_manual"], label="Train Cause", color=colors[2], linewidth=2)
    axes[1].plot(epochs, history[f"val_{target_name}_output_f1_score_manual"], label="Val Cause", color=colors[2], linestyle="dashed")
    
    axes[1].set_title(f"Pr Auc & F1 Score - {target_name.capitalize()} Outputs")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel('Pr Auc & F1 Score')
    axes[1].legend()
    axes[1].grid()
    
    # Saving and plotting
    plt.tight_layout()
    filename = f"{target_name}_composite.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

def plot_loss(epochs, history, output_dir):
    # Plot training and validation Loss
    plt.plot(epochs, history['loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    # Saving and plotting
    filename = f"loss_composite.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

@hydra.main(config_path="../../configs", config_name="main.yaml", version_base=None)
def main(cfg : DictConfig):

    # Load the history metrics
    with open(to_absolute_path(cfg.train.history_file), "r") as file:
        history = json.load(file)

    # Number of epochs (assuming all lists have the same length)
    epochs = range(1, len(history["components_output_mcc_manual"]) + 1)

    # Create the output directory if it doesn't exist
    output_dir = to_absolute_path(cfg.main.output_path)
    os.makedirs(output_dir, exist_ok=True)

    plot_composite_metric(epochs, history, output_dir, "components", ["#3357FF", "#E600AC", "#808000"]) 
    plot_composite_metric(epochs, history, output_dir, "type", ["#33FF57", "#00ccff", "#800000"])  
    plot_composite_metric(epochs, history, output_dir, "cause", ["#FF5733", "#ffcc00", "#008080"])  
    plot_loss(epochs, history, output_dir)



if __name__ == '__main__':
    main()