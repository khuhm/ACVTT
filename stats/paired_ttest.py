import os
import numpy as np
from scipy import stats
import argparse

def load_metric_values(directory, metric):
    values = []
    for file in sorted(os.listdir(directory)):
        if not file.endswith(".npy"):
            continue
        metric_path = os.path.join(directory, file)
        data = np.load(metric_path, allow_pickle=True).item()
        if metric in data:
            values.append(data[metric])
        else:
            print(f"Warning: '{metric}' not found in {file}")
    return np.array(values)

def paired_t_test(dir_a, dir_b, metric, output_file):
    values_a = load_metric_values(dir_a, metric)
    values_b = load_metric_values(dir_b, metric)

    if len(values_a) != len(values_b):
        raise ValueError("The number of samples in each directory must be equal.")

    t_stat, p_value = stats.ttest_rel(values_a, values_b)

    with open(output_file, 'w') as f:
        f.write(f"Paired t-test results for metric '{metric}':\n")
        f.write(f"T-statistic: {t_stat:.4f}\n")
        f.write(f"P-value: {p_value:.6f}\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired t-test between two sets of metric results.")
    parser.add_argument('--metric', type=str, required=True, help="Metric name (e.g., ssim, psnr)")
    parser.add_argument('--dir_a', type=str, required=True, help="Directory of method A results")
    parser.add_argument('--dir_b', type=str, required=True, help="Directory of method B results")
    parser.add_argument('--output_file', type=str, required=True, help="Output file to save results")
    args = parser.parse_args()

    paired_t_test(args.dir_a, args.dir_b, args.metric, args.output_file)
