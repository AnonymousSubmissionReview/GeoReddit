"""
This script generates state-level statistics on specific {topic} prevalence in Reddit activity from preprocessed CSV files
and creates US static choropleth maps visualizing this {topic} prevalence by ratios or abs count.

Input:
- An input folder containing:
    - p_{topic}_post.csv
        (total counts of {topic}-related posts per state and month)
    - p_state_counts_cumulative.csv
        (overall post and user counts per state; only geolocated users with ratio2 > 1)
    - o_2005-06to2023-12_filtered_authors.csv
        (list of geolocated users (ratio2 > 1) and their corresponding states)
    - p_{topic}_keyword.csv
        (counts of {topic}-related keywords per state and month)
    - p_{topic}_YYYY-MM.csv files
        (detailed {topic} post data per month; used to identify unique {topic} users)

Output:
- A CSV file saved in the output folder:
    - p_{topic}_static_info.csv
      (aggregated per-state statistics for the whole period including ratios and abs)
    - p_{topic}_static_info_abs.csv
      (aggregated per-state statistics for the whole period only including abs)


- A PNG file saved in the output folder:
    - p_{topic}_Static_State_Maps.png
      (choropleth maps illustrating {topic} prevalence based on posts ratio, users ratio, and average keywords per post)
    - p_{topic}_Static_State_Maps_abs.png
      (choropleth maps illustrating {topic} prevalence based on posts count, users count)

Columns in p_{topic}_static_info.csv:
- state: US state abbreviation (e.g., CA, TX, NY)
- post: Total number of {topic}-related posts in that state (geolocated users with ratio2 > 1)
- total_post: Total number of all posts in that state (geolocated users with ratio2 > 1)
- total_user: Total number of unique users in that state (geolocated users with ratio2 > 1)
- user: Number of unique users in that state who posted {topic}-related content
- term: Total count of {topic}-related keywords appearing in all {topic} posts in that state
- post_ratio: Percentage of {topic} posts among all posts in that state: {topic}_post / total_post * 100
- user_ratio: Percentage of {topic} users among all users in that state: {topic}_user / total_user * 100
- term_mean: Average number of {topic}-related keywords per {topic} post: {topic}_term / {topic}_post

Columns in p_{topic}_static_info_abs.csv:
- state: US state abbreviation (e.g., CA, TX, NY)
- post: Total number of {topic}-related posts in that state (geolocated users with ratio2 > 1)
- user: Number of unique users in that state who posted {topic}-related content

Example usage:

python a_04_static_map_table.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --topic AI
"""


import argparse
import pandas as pd
import os
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import sys

try:
    import kaleido
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaleido"])
import kaleido

#kaleido.get_chrome_sync()

# List of US state abbreviations
US_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
    'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN',
    'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
    'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA',
    'WV', 'WI', 'WY'
]

# This function loads topic-related files, aggregates statistics per state,
# calculates ratios and keyword mean counts, and saves them to a CSV summary.
def generate_static_info(input_folder, output_folder, topic="AI"):
    # Define filenames for input
    post_file = f"p_{topic}_post.csv"
    state_count_file = "p_state_counts_cumulative.csv"
    filtered_authors_file = "o_2005-06to2023-12_filtered_authors.csv"
    keywords_file = f"p_{topic}_keyword.csv"

    # Load per-state topic post counts
    post_df = pd.read_csv(os.path.join(input_folder, post_file))
    post_states = post_df.columns[1:]
    post_total = post_df[post_states].sum().reset_index()
    post_total.columns = ["state", "post"]

    # Load total posts and users per state
    state_count_df = pd.read_csv(os.path.join(input_folder, state_count_file))
    state_total = state_count_df[state_count_df["state"].isin(post_total["state"])]
    state_total = state_total.rename(
        columns={"2023_post": "total_post", "2023_user": "total_user"}
    )[["state", "total_post", "total_user"]]

    # Find all topic-related monthly files
    filtered_authors_df = pd.read_csv(os.path.join(input_folder, filtered_authors_file))
    pattern = rf"^p_{topic}_\d{{4}}-\d{{2}}\.csv$"
    _files = [f for f in os.listdir(input_folder) if re.match(pattern, f)]
    _users = set()
    # Collect all unique authors who posted about the topic
    for file in _files:
        df = pd.read_csv(os.path.join(input_folder, file))
        if "author" in df.columns and not df["author"].dropna().empty:
            _users.update(df["author"].dropna().unique())

    # Count unique authors per state
    users_df = filtered_authors_df[filtered_authors_df["author"].isin(_users)]
    user_counts = users_df.groupby("state")["author"].nunique().reset_index()
    user_counts.columns = ["state", "user"]
    user_counts = user_counts[user_counts["state"].isin(post_total["state"])]

    # Load keyword counts
    keywords_df = pd.read_csv(os.path.join(input_folder, keywords_file))
    terms_states = keywords_df.columns[1:]
    terms_total = keywords_df[terms_states].sum().reset_index()
    terms_total.columns = ["state", "term"]

    # Merge all statistics into one dataframe
    result = post_total.merge(state_total, on="state", how="inner") \
        .merge(user_counts, on="state", how="left") \
        .merge(terms_total, on="state", how="left")

    # Fill missing counts with 0
    result["user"] = result["user"].fillna(0).astype(int)
    result["term"] = result["term"].fillna(0).astype(int)

    # Calculate ratios and averages
    result["post_ratio"] = result["post"] / result["total_post"] * 100
    result["user_ratio"] = result["user"] / result["total_user"] * 100
    result["term_mean"] = result["term"] / result["post"]

    # Save output CSV
    output_file = os.path.join(output_folder, f"p_{topic}_static_info.csv")
    result.to_csv(output_file, index=False)
    print(f"{os.path.basename(output_file)} has been saved to: {output_file}")


#This function loads topic-related files, aggregates statistics per state,
# calculates only abs count, and saves them to a CSV summary.
def generate_abs_static_info(input_folder, output_folder, topic="AI"):
    # Define filenames for input
    post_file = f"p_{topic}_post.csv"
    filtered_authors_file = "o_2005-06to2023-12_filtered_authors.csv"

    # Load per-state topic post counts
    post_df = pd.read_csv(os.path.join(input_folder, post_file))
    post_states = post_df.columns[1:]
    post_total = post_df[post_states].sum().reset_index()
    post_total.columns = ["state", "post"]


    # Find all topic-related monthly files
    filtered_authors_df = pd.read_csv(os.path.join(input_folder, filtered_authors_file))
    pattern = rf"^p_{topic}_\d{{4}}-\d{{2}}\.csv$"
    _files = [f for f in os.listdir(input_folder) if re.match(pattern, f)]
    _users = set()
    # Collect all unique authors who posted about the topic
    for file in _files:
        df = pd.read_csv(os.path.join(input_folder, file))
        if "author" in df.columns and not df["author"].dropna().empty:
            _users.update(df["author"].dropna().unique())

    # Count unique authors per state
    users_df = filtered_authors_df[filtered_authors_df["author"].isin(_users)]
    user_counts = users_df.groupby("state")["author"].nunique().reset_index()
    user_counts.columns = ["state", "user"]
    user_counts = user_counts[user_counts["state"].isin(post_total["state"])]

    # Merge all statistics into one dataframe
    result = post_total.merge(user_counts, on="state", how="left")

    # Fill missing counts with 0
    result["user"] = result["user"].fillna(0).astype(int)

    # Save output CSV
    output_file = os.path.join(output_folder, f"p_{topic}_static_info_abs.csv")
    result.to_csv(output_file, index=False)
    print(f"{os.path.basename(output_file)} has been saved to: {output_file}")

# This function builds a single choropleth map for one metric
def create_choropleth_map(df, column, colorscale, colorbar_title, colorbar_x):
    return go.Choropleth(
        locations=df['state'],
        z=df[column].astype(float),
        locationmode='USA-states',
        colorscale=colorscale,
        autocolorscale=False,
        marker_line_width=0.5,
        colorbar=dict(
            title=dict(text=colorbar_title, side="bottom"),
            orientation="h",
            x=colorbar_x,
            xanchor="center",
            y=-0.08,
            thickness=8,
            len=0.25
        )
    )

# This function generates 3 side-by-side choropleth maps to visualize
# post ratio, user ratio, and mean keyword count per state.
def create_us_state_maps_for_ratios(output_folder, topic='AI'):
    # Load statistics CSV
    file_path = os.path.join(output_folder, f"p_{topic}_static_info.csv")
    df = pd.read_csv(file_path)

    # Create figure with 3 maps
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"{topic.capitalize()} Prevalence Based on Posts",
            f"{topic.capitalize()} Prevalence Based on Users",
            f"{topic.capitalize()} Prevalence Based on Terms"
        ],
        specs=[[{"type": "choropleth"}]*3],
        horizontal_spacing=0.02
    )

    # Add each choropleth map
    fig.add_trace(create_choropleth_map(df, 'post_ratio', 'reds',
                                        f"{topic.capitalize()}-Posts / Geolocated Posts (%)", colorbar_x=0.15), row=1, col=1)
    fig.add_trace(create_choropleth_map(df, 'user_ratio', 'blues',
                                        f"{topic.capitalize()}-Users / Geolocated Users (%)", colorbar_x=0.5), row=1, col=2)
    fig.add_trace(create_choropleth_map(df, 'term_mean', 'greens',
                                        f"{topic.capitalize()}-Term Count per {topic.capitalize()}-Post", colorbar_x=0.85), row=1, col=3)

    # General layout settings
    fig.update_layout(
        title={
            'text': f'{topic.capitalize()} Topic Prevalence State-Level Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16}
        },
        font=dict(family="Times New Roman", size=10),
        height=400,
        width=1000,
        margin=dict(t=100, b=100, l=100, r=100),
    )

    fig.update_geos(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)',
    )

    # Save the PNG image
    # output_image = os.path.join(output_folder, f"p_{topic}_Static_State_Maps.png")
    # fig.write_image(output_image, format="png", scale=3, width=1000, height=400)

    # run on HPC
    output_html = os.path.join(output_folder, f"p_{topic}_Static_State_Maps.html")
    fig.write_html(output_html, include_plotlyjs='cdn')

    print(f"Figure saved to: {output_image}")

    fig.show()

# This function generates 2 side-by-side choropleth maps to visualize post count and user count.
def create_us_state_maps_for_abs(output_folder, topic='AI'):
    # Load statistics CSV
    file_path = os.path.join(output_folder, f"p_{topic}_static_info_abs.csv")
    df = pd.read_csv(file_path)

    # Create figure with 3 maps
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"{topic.capitalize()} Prevalence Based on Posts",
            f"{topic.capitalize()} Prevalence Based on Users"
        ],
        specs=[[{"type": "choropleth"}] * 2],
        horizontal_spacing=0.02
    )

    # Move all subplot titles closer to their maps
    fig.update_annotations(yshift=-10)

    # Add each choropleth map
    fig.add_trace(create_choropleth_map(df, 'post', 'reds',
                                        f"{topic.capitalize()}-Posts Count", colorbar_x=0.245), row=1, col=1)
    fig.add_trace(create_choropleth_map(df, 'user', 'blues',
                                        f"{topic.capitalize()}-Users Count", colorbar_x=0.755), row=1, col=2)

    # General layout settings
    fig.update_layout(
        title={
            'text': f'{topic.capitalize()} Topic Prevalence State-Level Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16}
        },
        font=dict(family="Times New Roman", size=10),
        height=400,
        width=800,
        margin=dict(t=100, b=10, l=100, r=100),
    )

    fig.update_geos(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)',
    )

    # Save the PNG image
    # output_image = os.path.join(output_folder, f"p_{topic}_Static_State_Maps_abs.png")
    # fig.write_image(output_image, format="png", scale=3, width=1000, height=400)
    #
    # run on HPC
    output_html = os.path.join(output_folder, f"p_{topic}_Static_State_Maps_abs.html")
    fig.write_html(output_html, include_plotlyjs='cdn')

    print(f"Figure saved to: {output_image}")

    fig.show()

# Entry point to parse arguments and call the above functions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate topic statistics and create static choropleth maps.")
    parser.add_argument("--input_folder", required=True, help="Folder containing all input files.")
    parser.add_argument("--output_folder", required=True, help="Folder where output files will be saved.")
    parser.add_argument("--topic", required=True, help="Topic keyword (e.g., 'AI').")
    parser.add_argument("--abs", action="store_true", help="show absolute values instead of normalized ratios")

    args = parser.parse_args()

    if args.abs:
        generate_abs_static_info(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            topic=args.topic
        )

        create_us_state_maps_for_abs(
            output_folder=args.output_folder,
            topic=args.topic
        )

    else:
        generate_static_info(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            topic=args.topic
        )

        create_us_state_maps_for_ratios(
            output_folder=args.output_folder,
            topic=args.topic)
