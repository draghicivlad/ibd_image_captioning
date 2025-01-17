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


def compare_metrics_upgraded(input_folders, labels, title, output_folder):
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

    # Train loss
    plt.figure()

    index = 0
    for folder, df in metrics_data.items():
        data = df.dropna(subset=['train_loss'])
        x = data['step']
        y = data['train_loss']

        if not data.empty:
            plt.plot(x, y, label=labels[index])

        index += 1

    plt.title(f"{title} - train_loss")
    plt.xlabel("Step")
    plt.ylabel("train loss".title())
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, "train_loss.png")
    plt.savefig(plot_path)
    plt.close()
    plt.clf()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=False)

    metrics = {
        (0, 0): "val_loss",
        (0, 1): "val_perplexity",
        (1, 0): "val_bleu",
        (1, 1): "val_rouge",
    }

    for key, metric in metrics.items():
        index = 0
        for folder, df in metrics_data.items():
            data = df.dropna(subset=[metric])
            x = data['epoch']
            y = data[metric]

            if not data.empty:
                axs[key].plot(x, y, label=labels[index])

            index += 1

        axs[key].set_title(f"{title} - {metric}")

    for ax in axs.flat:
        ax.grid(True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(output_folder, "validation_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    plt.clf()


if __name__ == "__main__":
    # compare_metrics(["../outputs/a_en_lstm13_20_59", "../outputs/a_en_data_aug_lstm14_10_50"],
    #                 ["lstm", "lstm_aug"], "EN - LSTM", "../plots/en_lstm")
    #
    # compare_metrics(["../outputs/a_en_transformer14_00_41", "../outputs/a_en_data_aug_transformer14_17_19"],
    #                 ["transf", "transformer_aug"], "EN - Transformer", "../plots/en_transf")
    #
    # compare_metrics(["../outputs/ro_lstm_256_512_2_0.315_13_30", "../outputs/ro_transf_256_512_2_0.3"],
    #                 ["lstm", "transformer"], "Ro - LSTM vs Transformer", "../plots/ro_lstm_transf")
    #
    # compare_metrics(["../outputs/ro_transf_256_512_2_0.3", "../outputs/ro_transformer_transfer_lr"],
    #                 ["transformer", "transformer transfer"], "Ro - Transformer vs Transformer Transfer", "../plots/transf_transfer")

    compare_metrics_upgraded(["../outputs/a_en_lstm13_20_59", "../outputs/a_en_transformer14_00_41"],
                             ["lstm", "transformer"], "EN - LSTM vs Transformer", "../plots/en_lstm_transf")

    compare_metrics_upgraded(["../outputs/a_en_lstm13_20_59", "../outputs/a_en_data_aug_lstm14_10_50"],
                             ["lstm", "lstm_aug"], "EN - LSTM", "../plots/en_lstm")

    compare_metrics_upgraded(["../outputs/a_en_transformer14_00_41", "../outputs/a_en_data_aug_transformer14_17_19"],
                             ["transf", "transformer_aug"], "EN - Transformer", "../plots/en_transf")

    compare_metrics_upgraded(["../outputs/a_en_transformer14_00_41", "../outputs/distilled_pretrained_child_teacher"],
                             ["transf", "transformer_distil"], "EN - Transformer vs Distil vs Teacher", "../plots/en_transf_distil_second")

    compare_metrics_upgraded(["../outputs/ro_lstm_256_512_2_0.315_13_30", "../outputs/ro_transf_256_512_2_0.3"],
                             ["lstm", "transformer"], "RO - LSTM vs Transformer", "../plots/ro_lstm_transf")

    compare_metrics_upgraded(["../outputs/ro_transf_256_512_2_0.3", "../outputs/ro_transformer_256_512_2_0.3_transfer_learning_model16_20_15"],
                             ["transformer", "transformer transfer"], "RO - Transformer vs Transformer Transfer",
                             "../plots/ro_transf_transfer")
