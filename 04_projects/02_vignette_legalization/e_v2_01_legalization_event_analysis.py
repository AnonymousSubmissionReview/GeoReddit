r"""
This script computes the influence of legalizing marijuana on sentmiment words per state.
This script estimates linear-mixed models for sentiment by an event-variable while including random intercepts for states and accounting for general time trend.
The script computes a summary table for the standardized linear-mixed models, 
a barplot of the predicted unstandardized parameter estimates by event-variable,
and the z-standardized time trend of states which legalized marijuana ±12 months before and after the event variable.  
Furthermore the script generates a .png combining the summary table, barplot, and timetrend for both sentiments.

The script permutates across different data permutations:
- different base-rates for word share (proportion [prop]: the number of emotional words per post; frequency [freq]: the total number of words per post)
- different cut-off criteria of observations per state (n = 30, n = 50, and n = 100), 
- different event variables vote_012 (months before, month at, and months after voting to legalize marijuana) and poss_012 (months before, month at, and months after possession was officially legal), 
- different sentiments (i.e., positive and negative) 

Input:
- A base input folder containing:
    - Weed_combined_sentiment_per_author_month_n{obs_cutoffs}.csv; a .csv file created by a_09_sentiment_map_table.py
    - monthly_merged.csv; a .csv file containing the different event dummy codings

Output:
- A folder of the structure n{obs_cutoffs}_NRCL_{sentiment} including the summary output, barplot and timetrend for the different 
event_variables
- Figure plots which combine summary output, barplot and timetrend for all sentiments in one .png per 
observation cutoff, base-rates for word share, and event variable permutation: summary_grid_{base-rate}_n{obs_cutoffs}_{event_variable}

Example Usage:
python .../e_v2_01_legalization_event_analysis.py `
--input_folder "C:/.../Weed_maps" `
--output_folder "C:/.../Weed_event_analysis_barplot_linetrend" `
--dummy_file "C:/.../Dummy_coding_legalization/monthly_merged.csv"

"""

import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
from jinja2 import Template
import patsy
from matplotlib.image import imread
from tabulate import tabulate
from PIL import Image

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, required=True)
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--dummy_file', type=str, required=True)
args = parser.parse_args()

# === Configuration ===
obs_cutoffs = [30, 50, 100]
sentiments = ['NRCL_freq_positive', 'NRCL_freq_negative',
              'NRCL_prop_positive', 'NRCL_prop_negative']
aggregation = 'per_author'
agg_map = {'per_author': 'author'}
norm = 'zscore'
windows = [12, 24, 36]
event_vars = ['vote_012', 'poss_012']

# === Load dummy coding data ===
df_dummies = pd.read_csv(args.dummy_file)
df_dummies['time'] = pd.to_datetime(df_dummies['time'], format='%Y-%m')
df_dummies = df_dummies[df_dummies['time'] >= '2014-01-01']

# === Create output folder ===
os.makedirs(args.output_folder, exist_ok=True)

# === Helper Functions ===

def normalize_zscore(df):
    df['value'] = df.groupby('state')['value'].transform(lambda x: (x - x.mean()) / x.std())
    return df

def plot_aligned_trend(df, marker_col, headline, filename, window, output_path):
    relevant_states = df[df[marker_col] == 1]['state'].unique()
    df_filtered = df[df['state'].isin(relevant_states)].copy()
    event_dates = df_filtered[df_filtered[marker_col] == 1][['state', 'time']].drop_duplicates()
    event_dates = event_dates.rename(columns={'time': 'event_time'})
    df_filtered = df_filtered.merge(event_dates, on='state', how='left')
    df_filtered['rel_month'] = (df_filtered['time'].dt.year - df_filtered['event_time'].dt.year) * 12 + \
                               (df_filtered['time'].dt.month - df_filtered['event_time'].dt.month)
    df_filtered = df_filtered[(df_filtered['rel_month'] >= -window) & (df_filtered['rel_month'] <= window)]

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_filtered, x='rel_month', y='value', hue='state', estimator=None, lw=1, legend=False)
    sns.lineplot(data=df_filtered, x='rel_month', y='value', color='black', errorbar=('ci', 95), lw=2)
    plt.axvline(0, color='black', linestyle='--')
    plt.title(headline)
    plt.xlabel("Months Since Event")
    plt.ylabel("Sentiment/Emotion (z-score)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.close()

def fit_mixedlm(formula, df):
    model = smf.mixedlm(formula, df, groups=df["state"])
    fit = model.fit()
    return fit

def plot_event_margins_bar(df_model, event_var, filename, title, output_path):

    df_flat = df_model.reset_index()

    # Add standardized variables
    df_flat['value_z'] = (df_flat['value_raw'] - df_flat['value_raw'].mean()) / df_flat['value_raw'].std()
    df_flat['timec_z'] = (df_flat['timec'] - df_flat['timec'].mean()) / df_flat['timec'].std()

    # RAW MODEL (for barplot)
    formula_raw = f"value_raw ~ C({event_var}) + timec"
    model_raw = smf.mixedlm(formula_raw, data=df_flat, groups="state").fit()

    # STANDARDIZED MODEL (for summary table)
    formula_std = f"value_z ~ C({event_var}) + timec_z"
    model_std = smf.mixedlm(formula_std, data=df_flat, groups="state").fit()

    # PREDICTIONS using raw model
    ref_timec = df_flat['timec'].median()
    df_flat[event_var] = df_flat[event_var].astype('category')
    categories = df_flat[event_var].cat.categories

    pred_df = pd.DataFrame({event_var: categories, 'timec': ref_timec})
    pred_df[event_var] = pd.Categorical(pred_df[event_var], categories=categories)
    design_info = model_raw.model.data.design_info
    pred_design = patsy.build_design_matrices([design_info], pred_df)[0]
    pred = np.dot(pred_design, model_raw.fe_params)

    # Confidence intervals
    full_cov = model_raw.cov_params()
    fe_param_names = model_raw.fe_params.index
    cov_fe = full_cov.loc[fe_param_names, fe_param_names]
    pred_var = np.array([np.dot(row, np.dot(cov_fe, row)) for row in pred_design])
    pred_se = np.sqrt(pred_var)
    z = 1.96
    pred_df['predicted'] = pred
    pred_df['ci_lower'] = pred - z * pred_se
    pred_df['ci_upper'] = pred + z * pred_se

    # Coefficient summary from standardized model
    coefs_df = model_std.params.to_frame(name='Estimate')
    coefs_df['Std.Err.'] = model_std.bse
    coefs_df['t-value'] = model_std.tvalues
    coefs_df['p-value'] = model_std.pvalues
    coefs_df = coefs_df.reset_index().rename(columns={'index': 'Coefficient'})

    # Rename coefficients for clarity
    label_map = {
        'Intercept': 'intercept (k = 0)',
        f'C({event_var})[T.1]': f'{event_var} (k = 1)',
        f'C({event_var})[T.2]': f'{event_var} (k = 2)',
        'timec_z': 'timetrend (γ)',
        'state Var': 'state Var (bi)',
    }
    coefs_df['Coefficient'] = coefs_df['Coefficient'].replace(label_map)

    # Format values for table
    def format_estimate(val):
        return f"{val:.4f}"

    def format_stderr(val):
        return "<0.0001" if val < 0.0001 else f"{val:.4f}"

    def format_tval(val):
        return f"{val:.2f}"

    def format_pval(val):
        return "<.001" if val < 0.001 else f".{str(round(val, 3))[2:].zfill(3)}"

    coefs_df['Estimate'] = coefs_df['Estimate'].apply(format_estimate)
    coefs_df['Std.Err.'] = coefs_df['Std.Err.'].apply(format_stderr)
    coefs_df['t-value'] = coefs_df['t-value'].apply(format_tval)
    coefs_df['p-value'] = coefs_df['p-value'].apply(format_pval)

    # Save coefficients to CSV
    coefs_df.to_csv(os.path.join(output_path, f"{event_var}_coefs.csv"), index=False)

    # PLOTTING
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 2])

    # Table above plot
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis('off')
    table = ax_table.table(
        cellText=coefs_df.values,
        colLabels=coefs_df.columns,
        loc='center',
        cellLoc='center'
    )
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Barplot with CIs
    ax = fig.add_subplot(gs[1])
    sns.barplot(data=pred_df, x=event_var, y='predicted', ci=None, palette="muted", ax=ax)

    for i, row in pred_df.iterrows():
        ax.errorbar(
            x=i,
            y=row['predicted'],
            yerr=[[row['predicted'] - row['ci_lower']], [row['ci_upper'] - row['predicted']]], 
            fmt='none',
            c='black',
            capsize=5,
            lw=1.5
        )

    ax.set_title(title)
    ax.set_ylabel("Predicted Sentiment/Emotion")
    ax.set_xlabel(event_var.replace('_', ' ').capitalize())

    ymin = pred_df['ci_lower'].min()
    ymax = pred_df['ci_upper'].max()
    padding = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - padding, ymax + padding)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.close()


# === Main Loop ===
for n in obs_cutoffs:
    for senti in sentiments:
        filename_base = f"Weed_combined_sentiment_per_author_month_n{n}.csv"
        input_path = os.path.join(args.input_folder, filename_base) 

        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue

        df_all = pd.read_csv(input_path) 
        df_all['time'] = pd.to_datetime(df_all['period'], format='%Y-%m') 
        df_all = df_all[df_all['time'] >= '2014-01-01']

        # Select only the relevant sentiment column and rename it to 'value'
        df = df_all[['state', 'time', senti]].copy() 
        df = df.merge(df_dummies, on=['state', 'time'], how='left')
        df = df.rename(columns={senti: 'value'}) 
        df['value_raw'] = pd.to_numeric(df['value'], errors='coerce')  
        df = df.dropna(subset=['value_raw'])
        df = normalize_zscore(df)

        df['state'] = df['state'].astype('category')
        df['year_month'] = df['time'].dt.to_period('M').astype(str).astype('category')
        df['timec'] = (df['time'].dt.year - 2014) * 12 + (df['time'].dt.month - 1) + 1


        # === Output path for permutation ===
        permutation_folder = os.path.join(args.output_folder, f"n{n}_{senti}")
        os.makedirs(permutation_folder, exist_ok=True)

        # === Lineplots ===
        for window in windows:
            for marker_col in event_vars:
                plot_aligned_trend(
                    df,
                    marker_col=marker_col,
                    headline=f"{marker_col.upper()} ±{window} months — {senti} [n={n}]",
                    filename=f"{senti}_n{n}_{marker_col}_trend_{window}m.png",
                    window=window,
                    output_path=permutation_folder
                )

        # === Mixed Models and Bar Plots ===
        formulas = [
            "value_raw ~ 1 + C(vote_012) + timec",
            "value_raw ~ 1 + C(poss_012) + timec"
        ]
        model_names = ['vote_012 + trend', 'poss_012 + trend']
        model_files = ['vote_012', 'poss_012']
        models = []

        for formula in formulas:
            # Create standardized copy for modeling
            df_model = df.copy()
            df_model['value_z'] = (df_model['value_raw'] - df_model['value_raw'].mean()) / df_model['value_raw'].std()
            df_model['timec_z'] = (df_model['timec'] - df_model['timec'].mean()) / df_model['timec'].std()

            # Adjust formula to use standardized variables
            formulas = [
                "value_z ~ 1 + C(vote_012) + timec_z",
                "value_z ~ 1 + C(poss_012) + timec_z"
            ]

            # Fit models on standardized data
            models = []
            for formula in formulas:
                model = fit_mixedlm(formula, df_model.copy())  # standardized version
                models.append(model)

        # Save model summaries
        for name, model in zip(model_files, models):
            with open(os.path.join(permutation_folder, f"{name}_summary.txt"), "w") as f:
                f.write(model.summary().as_text())

        # Barplots
        for var in event_vars:
            plot_event_margins_bar(
                df_model=df,
                event_var=var,
                filename=f"predicted_sentiment_by_{var}.png",
                title=f"Predicted Sentiment by {var}",
                output_path=permutation_folder
            )

        # HTML Summary Report
        template = Template("""
        <html>
        <head>
            <title>Sentiment Analysis Report - {{ senti }} n{{ n }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2 { color: #2c3e50; }
                img { margin: 10px 0; border: 1px solid #ccc; width: 100%; max-width: 700px; }
                pre { background: #f7f7f7; padding: 15px; border-left: 4px solid #3498db; overflow-x: auto; white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <h1>Sentiment Analysis for {{ senti }} (n={{ n }})</h1>

            {% for window in windows %}
                {% for var in event_vars %}
                    <h2>Line Trend ±{{ window }} months — {{ var }}</h2>
                    <img src="{{ senti }}_n{{ n }}_{{ var }}_trend_{{ window }}m.png" alt="Lineplot" />
                {% endfor %}
            {% endfor %}

            {% for var in event_vars %}
                <h2>Bar Plot — Predicted Sentiment by {{ var }}</h2>
                <img src="predicted_sentiment_by_{{ var }}.png" alt="Barplot" />
            {% endfor %}

            <h2>Model Summaries</h2>
            {% for name, model in summaries.items() %}
                <h3>{{ name }}</h3>
                <pre>{{ model }}</pre>
            {% endfor %}
        </body>
        </html>
        """)

        html = template.render(
            senti=senti,
            n=n,
            windows=windows,
            event_vars=event_vars,
            summaries={name: model.summary().as_text() for name, model in zip(model_files, models)}
        )

        with open(os.path.join(permutation_folder, "report.html"), "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✅ Done with: {senti} | n={n}")

# === Create Combined 1x2 Plot (bar left, line right per sentiment) for selected sentiments ===

selected_sentiments = [s for s in sentiments] + [s.replace('freq', 'prop') for s in sentiments]

for n in obs_cutoffs:
    for event_var in event_vars:
        for senti in selected_sentiments:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))  # 1 row × 2 columns
            permutation_folder = os.path.join(args.output_folder, f"n{n}_{senti}")
            os.makedirs(permutation_folder, exist_ok=True)

            # === Barplot (left) ===
            barplot_file = os.path.join(permutation_folder, f"predicted_sentiment_by_{event_var}.png")
            print(f"Looking for barplot: {barplot_file}")
            if os.path.exists(barplot_file):
                img_bar = imread(barplot_file)
                axes[0].imshow(img_bar)
                axes[0].axis('off')
                axes[0].set_title(f"{senti} — Barplot", fontsize=10)
                axes[0].set_ylabel("Share of emotion words")
            else:
                axes[0].axis('off')
                axes[0].set_title(f"{senti} — Barplot NOT FOUND", fontsize=10)

            # === Lineplot (right) ===
            lineplot_file = os.path.join(permutation_folder, f"{senti}_n{n}_{event_var}_trend_12m.png")
            print(f"Looking for lineplot: {lineplot_file}")
            if os.path.exists(lineplot_file):
                img_line = imread(lineplot_file)
                axes[1].imshow(img_line)
                axes[1].axis('off')
                axes[1].set_title(f"{senti} — Lineplot (±12m)", fontsize=10)
                axes[1].set_ylabel("Share of emotion words (z-score)")
            else:
                axes[1].axis('off')
                axes[1].set_title(f"{senti} — Lineplot NOT FOUND", fontsize=10)

            plt.suptitle(f"Combined Plot — {senti} — n={n} — Event: {event_var}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            combined_filename = f"{senti}_line_and_bar_n{n}_{event_var}.png"
            combined_path = os.path.join(args.output_folder, combined_filename)
            plt.savefig(combined_path, dpi=300)
            plt.close()
            print(f"✅ Combined plot saved to: {combined_path}")


# === Generate Aggregated Summary Tables (one per freq/prop × n × event_var) ===

summary_sentiments = sentiments + [s.replace("freq", "prop") for s in sentiments]

for senti_type in ["freq", "prop"]:
    relevant_sentis = [s for s in summary_sentiments if senti_type in s]
    for n in obs_cutoffs:
        for event_var in event_vars:
            rows = []
            for senti in relevant_sentis:
                permutation_folder = os.path.join(args.output_folder, f"n{n}_{senti}")
                summary_file = os.path.join(permutation_folder, f"{event_var}_summary.txt")
                if not os.path.exists(summary_file):
                    continue

                with open(summary_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                coef_section = False
                for line in lines:
                    if "Coef." in line and "Std.Err." in line:
                        coef_section = True
                        continue
                    if coef_section:
                        if line.strip() == "":
                            break
                        parts = line.split()
                        if len(parts) >= 5:
                            coef_name = parts[0]
                            coef_val = parts[1]
                            stderr = parts[2]
                            tval = parts[3]
                            pval = parts[4]
                            rows.append({
                                "Sentiment": senti,
                                "Coefficient": coef_name,
                                "Estimate": coef_val,
                                "Std. Err.": stderr,
                                "t-value": tval,
                                "p-value": pval
                            })

            if rows:
                df_summary = pd.DataFrame(rows)

                # === CSV ===
                csv_filename = f"summary_{senti_type}_n{n}_{event_var}.csv"
                df_summary.to_csv(os.path.join(args.output_folder, csv_filename), index=False)

                # === HTML ===
                html_filename = f"summary_{senti_type}_n{n}_{event_var}.html"
                df_summary.to_html(os.path.join(args.output_folder, html_filename), index=False, border=0)

                # === TXT ===
                txt_filename = f"summary_{senti_type}_n{n}_{event_var}.txt"
                with open(os.path.join(args.output_folder, txt_filename), "w", encoding="utf-8") as f_txt:
                    f_txt.write(tabulate(df_summary, headers="keys", tablefmt="github", showindex=False))

                print(f"📄 Saved summaries: {csv_filename}, {html_filename}, {txt_filename}")


# === Final Summary Grids: One PNG per freq/prop × n × event_var ===
grid_rows, grid_cols = 2, 1  

for senti_type in ["freq", "prop"]:
    relevant_sentis = [s for s in sentiments if senti_type in s]
    
    for n in obs_cutoffs:
        for event_var in event_vars:
            images = []
            for senti in relevant_sentis:
                combined_file = os.path.join(args.output_folder, f"{senti}_line_and_bar_n{n}_{event_var}.png")
                if os.path.exists(combined_file):
                    try:
                        img = Image.open(combined_file)
                        images.append(img)
                    except Exception as e:
                        print(f"❌ Could not open image: {combined_file} — {e}")
                else:
                    print(f"⚠️ Combined image missing: {combined_file}")

            if not images:
                print(f"⚠️ No images found for type={senti_type}, n={n}, var={event_var}")
                continue

            # Get size from the first image
            width, height = images[0].size
            grid_w = grid_cols * width
            grid_h = grid_rows * height

            # Create new blank image
            summary_img = Image.new('RGB', (grid_w, grid_h), color='white')

            for idx, img in enumerate(images):
                row = idx // grid_cols
                col = idx % grid_cols
                x = col * width
                y = row * height
                summary_img.paste(img, (x, y))

            # Save final merged image
            summary_filename = f"summary_grid_{senti_type}_n{n}_{event_var}.png"
            summary_path = os.path.join(args.output_folder, summary_filename)
            summary_img.save(summary_path, dpi=(300, 300))
            print(f"🖼️ Saved grid summary: {summary_path}")
