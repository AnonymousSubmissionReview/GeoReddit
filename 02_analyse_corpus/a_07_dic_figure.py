"""
Generate comprehensive dictionary analysis visuals including distributions, top/bottom N words,
word clouds, trend lines, co-occurrence scatter, and hierarchical clustering dendrogram.

Inputs (in --input_folder):
  • p_keyword_totals_summary.csv
      – five meta rows + one row per phrase: [phrase, total, percentage]
  • p_keyword_degree_summary.csv
      – five meta rows + one row per phrase: [phrase, degree]
  • p_keyword_counts.csv
      – rows = months + “TOTAL”, columns = original phrases, values = counts
  • p_cooccurrence_edges.csv
      – columns = [Source, Target, Weight] for co-occurrence network

Outputs (in --output_folder):
-p_distribution.png
Two side-by-side histograms:
1. Total Mentions Distribution – the distribution of keyword total counts
2. Degree Distribution – the distribution of node degrees (number of co-occurrence partners) in the network

-p_Total Mentions_top_bottom_{n}.png
- Left: bar chart of the top N keywords by total mentions
- Right: bar chart of the bottom N keywords by total mentions

-p_Degree_top_bottom_{n}.png
- Left: the top N keywords by network degree (the most social hub terms)
- Right: the bottom N keywords by network degree (the most isolated long-tail terms)
Use: compare which concepts connect to many others and which remain on the fringe

-p_wordcloud_Total Mentions_top_{n}.png
Word cloud of the top N keywords by total mentions, highlighting the hottest head terms

-p_wordcloud_Total Mentions_bottom_{n}.png
Word cloud of the bottom N keywords by total mentions, showcasing the extreme long-tail terms

-p_wordcloud_Degree_top_{n}.png
Word cloud of the top N keywords by network degree, emphasizing the true hubs

-p_wordcloud_Degree_bottom_{n}.png
Word cloud of the bottom N keywords by network degree, showing the near-isolated nodes

-p_scatter.png
Log-Log Scatter Plot: Total Mentions vs. Network Degree
- X-axis: total mentions (log scale)
- Y-axis: node degree (log scale)
- Quadrant interpretive table:
    Quadrant | Features                    | Typical Meaning in AI Context                                            | Label
    I        | High mentions × High degree | Core hub terms that anchor broad discussions and link many topics         | Core Terms
    II       | High mentions × Low degree  | Hotspot isolates: very popular but topic-specific                          | Hotspot Isolates
    III      | Low mentions × High degree  | Bridge potentials: niche terms linking multiple areas                      | Bridge Potential Terms
    IV       | Low mentions × Low degree   | Peripheral niche terms: rare, isolated concepts                            | Niche Terms

-p_trend_top_{n}.png
Time-series line plot of the top N keywords by total mentions (linear scales)

-p_dendrogram.png
Hierarchical Clustering Dendrogram:
How: compute pairwise co-occurrence distances, run hierarchical (Ward) clustering, plot the dendrogram.
Why: keywords that merge at low heights co-occur frequently → they form a natural topic cluster.
Use: cut the tree at your chosen height to split keywords into meaningful subtopics without pre-defining any categories.

Example usage:
    python a_07_dic_figure.py --input_folder /path/to/data --output_folder /path/to/figures --top_n 50 --wc_n 200 --weight_thresh 1000
"""

import os
import argparse
import random
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.dates as mdates
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram


# Global style settings
def set_academic_style():
    """Configure academic-style plotting defaults"""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')  # modern matplotlib seaborn style
    except:
        plt.style.use('seaborn-whitegrid')       # fallback for older versions
    finally:
        plt.style.use('default')                 # ensure a base style is always applied

    # Update rcParams for consistent academic look
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'figure.facecolor': 'white',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'png'
    })


#Plot academic‐style distributions of total mentions and node degree.
def plot_distribution(totals, degree, out_folder):
    set_academic_style()

    # Create a figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Define academic color palette
    color1 = '#1f77b4'  # blue
    color2 = '#d62728'  # red

    # Compute summary statistics
    total_mean   = totals['total'].astype(float).mean()
    total_median = totals['total'].astype(float).median()
    degree_mean  = degree['degree'].astype(float).mean()
    degree_median= degree['degree'].astype(float).median()

    # Plot histogram for Total Mentions
    ax1.hist(
        totals['total'].astype(float),
        bins=50,
        color=color1,
        edgecolor='white',
        alpha=0.8
    )
    # Add mean and median vertical lines
    ax1.axvline(total_mean,   color='k', linestyle='--', linewidth=1,
                label=f'Mean: {total_mean:.1f}')
    ax1.axvline(total_median, color='k', linestyle=':',  linewidth=1,
                label=f'Median: {total_median:.1f}')
    # Set titles, labels, and log scale on y-axis
    ax1.set(
        title='(a) Distribution of Total Mentions',
        xlabel='Total Mentions',
        ylabel='Frequency',
        yscale='log'
    )
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(frameon=False, fontsize=8)

    # Plot histogram for Node Degree
    ax2.hist(
        degree['degree'].astype(float),
        bins=50,
        color=color2,
        edgecolor='white',
        alpha=0.8
    )
    # Add mean and median vertical lines
    ax2.axvline(degree_mean,   color='k', linestyle='--', linewidth=1,
                label=f'Mean: {degree_mean:.1f}')
    ax2.axvline(degree_median, color='k', linestyle=':',  linewidth=1,
                label=f'Median: {degree_median:.1f}')
    # Set titles, labels, and log scales on both axes
    ax2.set(
        title='(b) Distribution of Node Degree',
        xlabel='Degree',
        ylabel='Frequency',
        yscale='log',
        xscale='log'
    )
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(frameon=False, fontsize=8)

    # Adjust layout and save figure
    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(out_folder, 'p_distribution.png'))
    plt.close(fig)


# Draw top and bottom N horizontal bar charts for a metric
def plot_top_bottom(df, col, name, top_n, out_folder):
    # Apply academic style settings
    set_academic_style()

    # Ensure the column is float
    df[col] = df[col].astype(float)
    # Select top_n largest and smallest values
    top = df.nlargest(top_n, col).set_index('phrase')[col]
    bottom = df.nsmallest(top_n, col).set_index('phrase')[col]

    # Dynamically adjust figure height based on top_n
    height = max(4, top_n * 0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, height))

    # Choose font size for y-axis labels
    fontsize = max(6, 12 - top_n // 10)

    # Plot Top N bar chart
    top.plot.barh(ax=ax1, color='steelblue')
    ax1.invert_yaxis()  # highest values at top
    ax1.tick_params(labelsize=fontsize)
    ax1.set(title=f'Top {top_n} {name}', xlabel=name)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot Bottom N bar chart
    bottom.plot.barh(ax=ax2, color='orange')
    ax2.invert_yaxis()
    ax2.tick_params(labelsize=fontsize)
    ax2.set(title=f'Bottom {top_n} {name}', xlabel=name)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Final layout adjustments and save figure
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, f'p_{name}_top_bottom_{top_n}.png'))
    plt.close(fig)


# Draw an academic-style word cloud
def plot_wordcloud_top_bottom(df, col, name, wc_n, out_folder):
    set_academic_style()

    df[col] = df[col].astype(float)
    os.makedirs(out_folder, exist_ok=True)

    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                     '#bcbd22', '#17becf']

    def academic_color_func(word, *args, **kwargs):
        return random.choice(color_palette)

    # Top wordcloud
    top = df.nlargest(wc_n, col).set_index('phrase')[col].to_dict()
    wc_top = WordCloud(
        width=1200, height=600,
        background_color='white',
        max_words=wc_n,
        contour_width=1,
        contour_color='steelblue',
        colormap='viridis',
        prefer_horizontal=0.9,
        min_font_size=10,
        max_font_size=200,
        relative_scaling=0.5,
        color_func=academic_color_func
    ).generate_from_frequencies(top)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc_top, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Top {wc_n} {name} Terms', pad=20, fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.01, f"Word size represents relative frequency of {name.lower()}",
                ha='center', fontsize=10, style='italic')
    fig.savefig(os.path.join(out_folder, f'p_wordcloud_{name}_top_{wc_n}.png'))
    plt.close(fig)

    # Bottom wordcloud
    bottom = df.nsmallest(wc_n, col).set_index('phrase')[col].to_dict()
    wc_bottom = WordCloud(
        width=1200, height=600,
        background_color='white',
        max_words=wc_n,
        contour_width=1,
        contour_color='firebrick',
        colormap='plasma',
        prefer_horizontal=0.9,
        min_font_size=10,
        max_font_size=200,
        relative_scaling=0.5,
        color_func=academic_color_func
    ).generate_from_frequencies(bottom)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc_bottom, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Bottom {wc_n} {name} Terms', pad=20, fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.01, f"Word size represents relative frequency of {name.lower()}",
                ha='center', fontsize=10, style='italic')
    fig.savefig(os.path.join(out_folder, f'p_wordcloud_{name}_bottom_{wc_n}.png'))
    plt.close(fig)

# Draw the time series trend chart
def plot_trend(totals, counts_ts, top_n, out_folder):
    set_academic_style()

    totals['total'] = totals['total'].astype(float)
    top_phrases = totals.nlargest(top_n, 'total')['phrase']
    dates = pd.to_datetime(counts_ts.index, format='%Y-%m')

    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'x', '+', 'd', '|', '_']
    colors = plt.cm.tab20.colors + plt.cm.Set2.colors + plt.cm.Pastel1.colors

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, ph in enumerate(top_phrases):
        ax.plot(
            dates, counts_ts[ph],
            label=ph,
            color=colors[i % len(colors)],
            linestyle=linestyles[(i // len(markers)) % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=4,
            linewidth=1.5,
            alpha=0.8
        )


    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax.set(
        title=f'Trend of Top {top_n} Keywords by Total Mentions',
        xlabel='Month',
        ylabel='Mentions'
    )
    ax.legend(fontsize='small', ncol=2, framealpha=0.9)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, f'p_trend_top_{top_n}.png'))
    plt.close(fig)

# Plot a scatter plot with smart annotations (four quadrants)
def plot_scatter(totals, degree, out_folder):
    set_academic_style()

    df = pd.merge(totals[['phrase', 'total']],
                  degree[['phrase', 'degree']],
                  on='phrase').dropna()
    df['total'] = df['total'].astype(float)
    df['degree'] = df['degree'].astype(float)

    median_total = df['total'].median()
    median_degree = df['degree'].median()

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(
        df['degree'],
        df['total'],
        c='#1f77b4',
        alpha=0.5,
        edgecolors='white',
        linewidths=0.3,
        s=40
    )

    ax.set_xscale('log')
    ax.set_yscale('log')

    # annotation function that supports selecting in ascending order
    def annotate_representative(df, ax, condition, n=2, sort_by='composite',
                                ascending=False, offset=(10, 10), color='black', label=None):
        subset = df[condition].copy()

        if sort_by == 'composite':
            subset['score'] = subset['total'] * subset['degree']
            subset = subset.nlargest(n, 'score') if not ascending else subset.nsmallest(n, 'score')
        elif sort_by == 'total':
            subset = subset.nlargest(n, 'total') if not ascending else subset.nsmallest(n, 'total')
        elif sort_by == 'degree':
            subset = subset.nlargest(n, 'degree') if not ascending else subset.nsmallest(n, 'degree')
        elif sort_by == 'balance':
            subset = subset.nlargest(2 * n, 'total').nsmallest(n, 'degree')

        # Annotate selected terms
        for _, row in subset.iterrows():
            ax.annotate(row['phrase'],
                        (row['degree'], row['total']),
                        textcoords="offset points",
                        xytext=offset,
                        ha='center',
                        fontsize=9,
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white',
                                  alpha=0.9,
                                  edgecolor='lightgray'),
                        arrowprops=dict(arrowstyle='->',
                                        color=color,
                                        alpha=0.6))
        if label:
            ax.text(0.98 if offset[0] > 0 else 0.02,
                    0.98 if offset[1] > 0 else 0.02,
                    label, transform=ax.transAxes,
                    ha='right' if offset[0] > 0 else 'left',
                    va='top' if offset[1] > 0 else 'bottom',
                    color=color, fontweight='bold')

    # 1. Core Terms-right up
    annotate_representative(df, ax,
                            condition=(df['degree'] > median_degree) &
                                      (df['total'] > median_total),
                            n=3, sort_by='composite',
                            offset=(15, 15), color='#d62728',
                            label='Core Terms')

    # 2. Bridge Potential Terms-right down
    annotate_representative(df, ax,
                            condition=(df['degree'] > median_degree) &
                                      (df['total'] <= median_total),
                            n=3, sort_by='total',
                            ascending=True,
                            offset=(15, -15), color='#2ca02c',
                            label='Bridge Potential Terms')

    # 3. Hotspot Isolates-left up
    annotate_representative(df, ax,
                            condition=(df['degree'] <= median_degree) &
                                      (df['total'] > median_total),
                            n=3, sort_by='degree',
                            ascending=True,
                            offset=(-15, 15), color='#9467bd',
                            label='Hotspot Isolates')

    # 4. lower_left
    lower_left = df[(df['degree'] <= median_degree) &
                    (df['total'] <= median_total)].copy()

    if not lower_left.empty:
        # Method 1: select the extreme cases with the lowest absolute values
        special_cases = lower_left.nsmallest(3, ['total', 'degree'])

        # Method 2: or select relatively prominent points (based on z-score from mean and standard deviation)
        lower_left['z_score'] = (
                (lower_left['total'] - lower_left['total'].mean()) / lower_left['total'].std() +
                (lower_left['degree'] - lower_left['degree'].mean()) / lower_left['degree'].std()
        )
        special_cases = lower_left.nsmallest(3, 'z_score')

        # Dynamically adjust the number of annotations (at least one)
        n = min(3, len(special_cases))
        special_cases = special_cases.head(n)

        # Special annotation style (larger font size and arrows)
        for _, row in special_cases.iterrows():
            ax.annotate(f"↓ {row['phrase']}",
                        (row['degree'], row['total']),
                        textcoords="offset points",
                        xytext=(random.randint(-25, 25), random.randint(-25, 25)),
                        ha='center',
                        fontsize=10,  # 加大字号
                        color='#ff7f0e',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white',
                                  edgecolor='#ff7f0e',
                                  alpha=0.9),
                        arrowprops=dict(arrowstyle='->',
                                        linewidth=1.5,  # 加粗箭头
                                        color='#ff7f0e',
                                        alpha=0.8))

        ax.text(0.02, 0.02,
                'Niche Terms:\n' + '\n'.join(special_cases['phrase'].tolist()),
                transform=ax.transAxes,
                ha='left',
                va='bottom',
                color='#ff7f0e',
                fontsize=9,
                bbox=dict(facecolor='white', edgecolor='#ff7f0e', alpha=0.8))
    else:
        print("No data points in the lower-left quadrant; skipping annotation")

    ax.axhline(median_total, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(median_degree, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set(title='Keyword Analysis: Mentions vs. Co-occurrence',
           xlabel='Degree in Co-occurrence Network (log scale)',
           ylabel='Total Mentions in AI Context (log scale)')
    ax.grid(True, linestyle=':', alpha=0.4)

    fig.tight_layout()
    os.makedirs(out_folder, exist_ok=True)
    fig.savefig(os.path.join(out_folder, 'p_scatter.png'))
    plt.close(fig)

# Draw a hierarchical clustering dendrogram of keywords
def plot_dendrogram(edge_csv, weight_thresh, out_folder,figsize=(14, 8)):
    set_academic_style()

    edges = pd.read_csv(edge_csv)
    filtered = edges[edges['Weight'] >= weight_thresh]
    terms = pd.unique(filtered[['Source', 'Target']].values.ravel())

    # 2. Build co‐occurrence matrix
    n = len(terms)
    if n == 0:
        raise ValueError(f"No terms with weight ≥ {weight_thresh}")

    matrix = pd.DataFrame(np.zeros((n, n)), index=terms, columns=terms)
    for _, row in filtered.iterrows():
        s, t, w = row['Source'], row['Target'], row['Weight']
        matrix.at[s, t] = matrix.at[t, s] = w

    # 3. clustering
    dist_arr = pdist(matrix.values, metric='cosine')
    Z = linkage(dist_arr, method='ward')

    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(Z,
               labels=matrix.index.tolist(),
               leaf_rotation=90,
               leaf_font_size=8,
               orientation='top',
               ax=ax)

    ax.set_title(f'Keyword Clustering Dendrogram (Weight ≥ {weight_thresh})')
    ax.set_ylabel('Ward Distance')
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "p_dendrogram.png"))
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate dictionary analysis figures")
    parser.add_argument("-i", "--input_folder", required=True, help="Folder with input CSVs and edges")
    parser.add_argument("-o", "--output_folder", required=True, help="Directory to save output PNGs")
    parser.add_argument("-n", "--top_n", type=int, default=50, help="Number of top terms to plot")
    parser.add_argument("--wc_n", type=int, default=200, help="Max words in word cloud")
    parser.add_argument("--weight_thresh", type=int, default=1000, help="Min edge weight for dendrogram")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    totals = pd.read_csv(
        os.path.join(args.input_folder, "p_keyword_totals_summary.csv"),
        header=0, skiprows=range(1, 6), encoding='utf-8-sig'
    )
    degree = pd.read_csv(
        os.path.join(args.input_folder, "p_keyword_degree_summary.csv"),
        header=0, skiprows=range(1, 6), encoding='utf-8-sig'
    )
    counts = pd.read_csv(
        os.path.join(args.input_folder, "p_keyword_counts.csv"),
        encoding='utf-8-sig'
    )
    counts_ts = counts[counts['month'] != 'TOTAL'].set_index('month')
    edge_csv = os.path.join(args.input_folder, "p_cooccurrence_edges.csv")


    plot_distribution(totals, degree, args.output_folder)
    plot_top_bottom(totals, 'total', 'Total Mentions', args.top_n, args.output_folder)
    plot_top_bottom(degree, 'degree', 'Degree', args.top_n, args.output_folder)
    plot_wordcloud_top_bottom(totals, 'total', 'Total Mentions', args.wc_n, args.output_folder)
    plot_wordcloud_top_bottom(degree, 'degree', 'Degree', args.wc_n, args.output_folder)
    plot_scatter(totals, degree, args.output_folder)
    plot_trend(totals, counts_ts, args.top_n, args.output_folder)
    plot_dendrogram(edge_csv, args.weight_thresh,args.output_folder)

    print(f"All figures saved to: {args.output_folder}")


if __name__ == "__main__":
    main()