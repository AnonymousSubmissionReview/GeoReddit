r"""
This script generates correlations between sentiment change (i.e, change in anger) and various regional, state-level covariates, and visualizes them in a forest plot. 
In addition, this script generates a scatterplot for the correlation between the outcome variable and voting Republican in the 2016 presidential election. 
Plots vary by outcome variable, cutoff, and correlation estimation method.


Input:
- A base input folder containing:
    - {topic}_{outcome_variable}_author_event_diff_summary.csv; the csv output files created by script v1_02_pepsi_event_change_map.py
    - State_covariates.csv; a .csv file containing the different regional covariates

Output:
- A .png displaying on the left handside a forest plot and on the right handside the scatterplot with republican voting behavior for the presidential election in 2016:
    - {topic}_{outcome_variable}_correlation_forest_and_scatter_n{threshold}_{corr_method}.png"
      (all matched posts with detailed metadata)

## Pepsi

Example Usage:
python .../e_v1_03_pepsi_regional_covariates_forest_plot_with_scatterplot.py `
--input_folder "C:/.../Pepsi_regional_covariates" `
--output_folder "C:/.../Pepsi_regional_correlation" `
--topic Pepsi

"""



import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, spearmanr, t
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FuncFormatter

def bootstrap_ci(x, y, method='pearson', n_bootstrap=2000, ci=95, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(x)
    bootstrapped_corrs = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        sample_x = x[indices]
        sample_y = y[indices]

        if method == 'pearson':
            r, _ = pearsonr(sample_x, sample_y)
        elif method == 'spearman':
            r, _ = spearmanr(sample_x, sample_y)
        else:
            raise ValueError("Invalid method. Choose 'pearson' or 'spearman'.")

        bootstrapped_corrs.append(r)

    lower_bound = np.percentile(bootstrapped_corrs, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_corrs, 100 - (100 - ci) / 2)

    # Final correlation (not bootstrapped)
    if method == 'pearson':
        r, _ = pearsonr(x, y)
    else:
        r, _ = spearmanr(x, y)

    return r, lower_bound, upper_bound

def plot_regression_line_with_ci(x, y, ax):
    x_reshape = x.reshape(-1, 1)
    model = LinearRegression().fit(x_reshape, y)
    y_pred = model.predict(x_reshape)
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    residuals = y - y_pred
    dof = len(x) - 2
    residual_std_error = np.sqrt(np.sum(residuals ** 2) / dof)
    t_val = t.ppf(0.975, dof)

    se_pred = residual_std_error * np.sqrt(
        1 / len(x) + (x_sorted - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2)
    )

    lower = y_pred_sorted - t_val * se_pred
    upper = y_pred_sorted + t_val * se_pred

    ax.plot(x_sorted, y_pred_sorted, color='red', label="Regression line")
    ax.fill_between(x_sorted, lower, upper, color='red', alpha=0.2, label="95% CI")
    ax.legend()

def main(input_folder, output_folder, topic="Pepsi"):

    # Covariates
    var_list = [
        "Poverty_Level_Estimate", "MedianHHinc", "Percent_Male", "Percent_White",
        "Percent_Black", "Percent_high_school", "Percent_Bachelor", "Percent_republican",
        "extra", "sci", "stabil", "pop_dens", "open", "agree"
    ]

    # Rename map
    label_map = {
        "Poverty_Level_Estimate": "Poverty Level",
        "MedianHHinc": "Median Household Income",
        "Percent_Male": "% Male Population",
        "Percent_White": "% White Population",
        "Percent_Black": "% Black Population",
        "Percent_high_school": "% High School Graduates",
        "Percent_Bachelor": "% Bachelor’s Degree Graduates",
        "Percent_republican": "% Voting Republican",
        "extra": "Big Five: Extraversion",
        "sci": "Big Five: Conscientiousness",
        "stabil": "Big Five: Emotional Stability",
        "pop_dens": "Population Density",
        "open": "Big Five: Openness",
        "agree": "Big Five: Agreeableness"
    }

    outcome_vars = ["NRCL_freq_anger", "NRCL_prop_anger"]

    for outcome_variable in outcome_vars:  
        # Load data
        sentiment_path = os.path.join(input_folder, f"{topic}_{outcome_variable}_author_event_diff_summary.csv")  
        covariates_path = os.path.join(input_folder, "State_covariates.csv")

        sentiment_df = pd.read_csv(sentiment_path)
        covariates_df = pd.read_csv(covariates_path)

        sentiment_df['state'] = sentiment_df['state'].str.upper()
        covariates_df['state'] = covariates_df['state'].str.upper()

        merged_df = pd.merge(sentiment_df, covariates_df, on='state', how='inner')

        # Remove DC entirely from dataset
        merged_df = merged_df[merged_df["state"] != "DC"]    

        for threshold in [30, 50]:
            clean_df = merged_df[merged_df[f"n_event1_author_{outcome_variable}"] >= threshold].copy()

            for corr_method in ['pearson', 'spearman']:
                results = []
                for var in var_list:
                    if var not in clean_df.columns:
                        continue
                    subset = clean_df[[var, f"diff_author_{outcome_variable}"]].dropna()
                    if subset.shape[0] < 3:
                        continue
                    x = subset[f"diff_author_{outcome_variable}"].to_numpy()
                    y = subset[var].to_numpy()
                    r, lower, upper = bootstrap_ci(x, y, method=corr_method, n_bootstrap=2000, ci=95, random_state=42)
                    results.append((var, r, lower, upper))

                forest_df = pd.DataFrame(results, columns=["Variable", "Correlation", "CI_Lower", "CI_Upper"])
                forest_df = forest_df.sort_values("Correlation")

                # Scatter plot data (always for Pearson Republican, full data)
                scatter_subset = clean_df[
                    [f"diff_author_{outcome_variable}", "Percent_republican", "state", f"n_event1_author_{outcome_variable}"]
                ].dropna()
                x = scatter_subset[f"diff_author_{outcome_variable}"].to_numpy()
                y = scatter_subset["Percent_republican"].to_numpy()
                r, lower, upper = bootstrap_ci(x, y, method=corr_method, n_bootstrap=2000, ci=95, random_state=42)

                # Create side-by-side plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(forest_df) * 0.35)))

                # Forest Plot
                sns.set(style="whitegrid")
                ax1.errorbar(
                    x=forest_df["Correlation"],
                    y=range(len(forest_df)),
                    xerr=[
                        forest_df["Correlation"] - forest_df["CI_Lower"],
                        forest_df["CI_Upper"] - forest_df["Correlation"]
                    ],
                    fmt='s',  # black square
                    color='black',
                    ecolor='gray',
                    capsize=4
                )
                ax1.set_yticks(range(len(forest_df)))
                ax1.set_yticklabels([label_map.get(v, v) for v in forest_df["Variable"]])
                ax1.axvline(0, color='black', linestyle='--')
                ax1.set_xlabel(f"{corr_method.capitalize()} Correlation (95% CI)")
                ax1.set_title(f"Forest Plot (n ≥ {threshold}) - {corr_method.capitalize()}")
                ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".lstrip('0').replace("-0", "-")))

                # Scatter Plot
                sns.scatterplot(
                    data=scatter_subset,
                    x=f"diff_author_{outcome_variable}",
                    y="Percent_republican",
                    size=f"n_event1_author_{outcome_variable}",
                    sizes=(20, 200),
                    color="steelblue",
                    legend=False,
                    ax=ax2
                )
                for _, row in scatter_subset.iterrows():
                    ax2.text(
                        row[f"diff_author_{outcome_variable}"],
                        row["Percent_republican"],
                        row["state"],
                        fontsize=8,
                        alpha=0.7,
                        verticalalignment='bottom',
                        horizontalalignment='right'
                    )
                plot_regression_line_with_ci(x, y, ax2)
                ax2.set_title(f"Scatter: % Republican vs. diff_author_{outcome_variable}\n{corr_method} r = {r:.2f}, 95% CI = [{lower:.2f}, {upper:.2f}]")
                ax2.set_xlabel("Change in share of anger-related words")
                ax2.set_ylabel("% Voting Republican")

                # Add correlation info as a box in lower left corner
                corr_text = (
                    f"$\\it{{r}}$ = {r:.2f}".lstrip("0").replace("-0", "-") + "\n"
                    + f"95% CI = [{lower:.2f}, {upper:.2f}]".replace("-0", "-").replace(" 0", " .") 
                )
                corr_text = corr_text.replace("0.", ".").replace("[.", "[.").replace(", .", ", .") 
                ax2.text(
                    0.02, 0.02, corr_text,
                    transform=ax2.transAxes,
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
                ) 

                plt.tight_layout()
                output_path = os.path.join(
                    output_folder,
                    f"{topic}_{outcome_variable}_correlation_forest_and_scatter_n{threshold}_{corr_method}.png"
                )
                plt.savefig(output_path, dpi=300)
                plt.close()
                print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate correlation forest and scatter plots.")
    parser.add_argument("--input_folder", required=True, help="Input folder with CSVs.")
    parser.add_argument("--output_folder", required=True, help="Output folder for plot.")
    parser.add_argument("--topic", default="Pepsi", help="Topic name (default: Pepsi)")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    main(args.input_folder, args.output_folder, args.topic)