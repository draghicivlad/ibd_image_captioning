import os
import pandas as pd
import matplotlib.pyplot as plt

def compare_metrics(input_folders, labels, title, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metrics_data = {}

    for folder in input_folders:
        metrics_file = os.path.join(folder, 'logs', 'lightning_logs', 'version_0', 'metrics.csv')
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            metrics_data[folder] = df
        else:
            print(f"Warning: {metrics_file} not found.")

    if not metrics_data:
        print("No valid metrics.csv files found.")
        return

    metrics = ['train_loss', 'val_bleu', 'val_loss', 'val_perplexity', 'val_rouge']

    for metric in metrics:
        plt.figure()

        index = 0
        for folder, df in metrics_data.items():
            if metric == 'train_loss':
                data = df.dropna(subset=['train_loss'])
                x = data['step']
                y = data['train_loss']
            else:
                data = df.dropna(subset=[metric])
                x = data['epoch']
                y = data[metric]

            if not data.empty:
                plt.plot(x, y, label=labels[index])

            index += 1

        plt.title(f"{title} - {metric}")
        plt.xlabel("Step")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_folder, f"{metric}.png")
        plt.savefig(plot_path)
        plt.close()

if __name__ == "__main__":
    compare_metrics(["../outputs/a_en_lstm13_20_59", "../outputs/a_en_data_aug_lstm14_10_50"],
                    ["lstm", "lstm_aug"], "EN - LSTM", "../plots/en_lstm")

    compare_metrics(["../outputs/a_en_transformer14_00_41", "../outputs/a_en_data_aug_transformer14_17_19"],
                    ["transf", "transformer_aug"], "EN - Transformer", "../plots/en_transf")

    compare_metrics(["../outputs/ro_lstm_256_512_2_0.315_13_30", "../outputs/ro_transf_256_512_2_0.3"],
                    ["lstm", "transformer"], "Ro - LSTM vs Transformer", "../plots/ro_lstm_transf")

    compare_metrics(["../outputs/ro_transf_256_512_2_0.3", "../outputs/ro_transformer_transfer_lr"],
                    ["transformer", "transformer transfer"], "Ro - Transformer vs Transformer Transfer", "../plots/transf_transfer")