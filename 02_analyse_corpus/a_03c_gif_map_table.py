"""
This script generates an interactive dynamic choropleth map to visualize topic prevalence across US states over time (monthly, quarterly, or yearly aggregation).
Specifically, it sets the color bar range using the 95th percentile to prevent extreme monthly values from distorting the color scale,
ensuring clearer and more balanced visual comparisons over time.


Input:
- An input folder containing:
    - p_{topic}_{data_type}_{freq}.csv
        (e.g., p_AI_author.csv, p_AI_post.csv, p_AI_keyword.csv; containing counts of matched posts or authors per state and month)
    - p_{topic}_norm_{data_type}_{freq}.csv
        (e.g., p_post_counts_per_month.csv, p_author_counts_per_month.csv; containing normalization results)

Output:
- An HTML file saved in the output folder:
    - p_{topic}_{data_type}_gifmap_{freq}.html
        (animated choropleth map showing normalized topic prevalence)
    or - p_{topic}_{data_type}_gifmap_{freq}_abs.html
    (animated choropleth map showing absolute topic prevalence)

Example usage:

# Example 1: Generate a monthly choropleth map of AI posts
python a_03_gif_map_table.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --topic AI --data_type post --freq month

# Example 2: Generate a quarterly choropleth map of AI unique authors
python a_03_gif_map_table.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --topic AI --data_type author --freq quarter

# Example 3: Generate a yearly choropleth map of keyword occurrences
python a_03c_gif_map_table.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --topic AI --data_type keyword --freq year
"""

import plotly.express as px
import pandas as pd
import argparse
import os

#Generate a dynamic choropleth map (GIF-style in HTML) showing either normalized ratios
#or absolute values of topic prevalence across US states over time.
def generate_topic_choropleth(input_folder, output_folder, topic="AI", data_type="author", freq="month", abs=False):
    # Input files
    if abs:
        data_file = os.path.join(input_folder, f"p_{topic}_{data_type}_{freq}.csv")
        output_map_file = os.path.join(output_folder, f"p_{topic}_{data_type}_gifmap_{freq}_abs.html")
    else:
        data_file = os.path.join(input_folder, f"p_{topic}_norm_{data_type}_{freq}.csv")
        output_map_file = os.path.join(output_folder, f"p_{topic}_{data_type}_gifmap_{freq}.html")


    df = pd.read_csv(data_file)

    df_long = pd.melt(df, id_vars=["time"], var_name="state", value_name="value")
    color_title = {
            "author": f"{topic.capitalize()} Geo-Users / All Geo-Users",
            "post": f"{topic.capitalize()} Geo-Posts / All Geo-Posts"
        }[data_type]

    # Color scale range
    #method1: used for data with little outliers
    #mean = df_long["value"].mean()
    #std_dev = df_long["value"].std()
    #upper_bound = mean + 3 * std_dev

    # #method2: Remove outliers
    upper_bound = df_long["value"].quantile(0.95)

    # Determine suffix for map title
    title_suffix = (
        "Users" if data_type == "author" else
        "Posts"
    )

    # Plot map
    fig = px.choropleth(
        df_long,
        locations='state',
        locationmode="USA-states",
        color="value",
        animation_frame="time",
        color_continuous_scale='Reds',
        range_color=(0, upper_bound),
        scope="usa",
        labels={"value": color_title},
    )

    fig.update_layout(
        title={
            'text': f'{topic.capitalize()} Topic Prevalence Based on Geolocated Reddit {title_suffix} ({freq.capitalize()} Aggregation)',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'Times New Roman, serif', 'size': 20, 'color': 'black'}
        },
        mapbox_style='carto-positron',
        width=800,
        height=600,
        margin={'r':0, 't':0, 'l':0, 'b':0}
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(
                text=color_title,
                font=dict(family='Times New Roman, serif', size=12, color='black')
            ),
            tickfont=dict(family='Times New Roman, serif', size=12, color='black')
        )
    )

    # Slider formatting
    fig.layout.sliders[0].currentvalue.xanchor = 'left'
    fig.layout.sliders[0].currentvalue.offset = -100
    fig.layout.sliders[0].currentvalue.prefix = ''
    fig.layout.sliders[0].len = .9
    fig.layout.sliders[0].currentvalue.font.color = 'indianred'
    fig.layout.sliders[0].currentvalue.font.size = 20
    fig.layout.sliders[0].y = 1.1
    fig.layout.sliders[0].x = 0.15
    fig.layout.updatemenus[0].y = 1.27
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000

    fig.write_html(output_map_file)
    print(f"Dynamic choropleth map saved to {output_map_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dynamic topic choropleth maps with monthly, quarterly, or yearly aggregation."
    )
    parser.add_argument("--input_folder", required=True, help="Path to folder containing input CSV files.")
    parser.add_argument("--output_folder", required=True, help="Path to folder where output files will be saved.")
    parser.add_argument("--topic", required=True, help="Topic keyword (e.g., 'AI').")
    parser.add_argument("--data_type", choices=["author", "post"], default="author", help="Type of data to visualize.")
    parser.add_argument("--freq", choices=["month", "quarter", "year"], default="month", help="Time aggregation frequency.")
    parser.add_argument("--abs", action="store_true", help="show absolute values instead of normalized ratios")
    args = parser.parse_args()

    generate_topic_choropleth(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        topic=args.topic,
        data_type=args.data_type,
        freq=args.freq,
        abs=args.abs
    )
