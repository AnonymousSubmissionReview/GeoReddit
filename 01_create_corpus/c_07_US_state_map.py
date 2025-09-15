"""
This script generates side-by-side choropleth maps of U.S. states to visualize the relationship between state-level population and geolocated Reddit activity over a specified period.

Input:
- An input folder containing:
    - p_state_counts_cumulative.csv
    - o_census.csv

Output:
- A PNG image saved in the output folder:
    - p_YYYY-MMtoYYYY-MM_US_State_Maps.png
      This image contains:
        1. Population by state
        2. Geolocated Reddit user counts by state
        3. Geolocated Reddit post counts by state

Example usage:
python c_07_US_state_maps.py --input_folder "C:/Users/u2288/Downloads/period" --output_folder "C:/Users/u2288/Downloads/period" --start_year 2005 --start_month 6 --end_year 2023 --end_month 12

"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import argparse

# Create a choropleth layer for a specified data column
def create_choropleth_map(df, column, colorscale, colorbar_title, colorbar_x):
    return go.Choropleth(
        locations=df['state'],
        z=df[column].astype(float),
        locationmode='USA-states',
        colorscale=colorscale,
        autocolorscale=False,
        marker_line_width=0.5,
        colorbar=dict(
            title=colorbar_title,
            orientation="h",
            x=colorbar_x,
            xanchor="center",
            y=-0.08,
            thickness=8,
            len=0.25
        )
    )

# Generate and save the state-level choropleth maps
def create_us_state_maps(input_folder, output_folder, start_year, start_month, end_year, end_month, year):
    # Build the input CSV file path with p_ prefix
    file_1 = "p_state_counts_cumulative.csv"
    file_path_1 = os.path.join(input_folder, file_1)
    df_1 = pd.read_csv(file_path_1)
    file_2 = "o_census.csv"
    file_path_2 = os.path.join(input_folder, file_2)
    df_2 = pd.read_csv(file_path_2)

    # Initialize the figure with 3 subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"U.S. State {year} Resident Population",
            "Reddit Geolocated Users by State",
            "Reddit Geolocated Posts by State"
        ],
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}, {"type": "choropleth"}]],
        horizontal_spacing=0.02
    )

    users_col = f"{end_year}_user"
    posts_col = f"{end_year}_post"
    pop_col = f"{year}"
    # Add population choropleth
    fig.add_trace(create_choropleth_map(df_2, pop_col, 'reds', "Population", colorbar_x=0.15), row=1, col=1)
    # Add user count choropleth
    fig.add_trace(create_choropleth_map(df_1, users_col, 'blues', "Users", colorbar_x=0.5), row=1, col=2)
    # Add post count choropleth
    fig.add_trace(create_choropleth_map(df_1, posts_col, 'greens', "Posts", colorbar_x=0.85), row=1, col=3)

    # Configure figure layout
    fig.update_layout(
        title={
            'text': f'U.S. State-Level Analysis ({start_year}.{start_month} - {end_year}.{end_month})',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16}
        },
        font=dict(family="Times New Roman", size=10),
        height=200,
        width=500,
        margin=dict(t=100, b=100, l=50, r=50),

    )

    # Configure the geographic projection and lakes
    fig.update_geos(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'
    )

    # Build output file path with p_ prefix
    output_file = os.path.join(
        output_folder,
        f"p_{start_year}-{str(start_month).zfill(2)}to{end_year}-{str(end_month).zfill(2)}_US_State_Maps.png"
    )

    # Save figure as PNG
    fig.write_image(output_file, format="png", scale=3,width=900, height=400)
    print(f"Figure saved to: {output_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate choropleth maps of U.S. state population, Reddit user counts, and Reddit post counts."
    )
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing the CSV file.")
    parser.add_argument("--output_folder", required=True, help="Path to the folder to save the output image.")
    parser.add_argument("--start_year", type=int, required=True, help="Start year.")
    parser.add_argument("--start_month", type=int, required=True, help="Start month.")
    parser.add_argument("--end_year", type=int, required=True, help="End year.")
    parser.add_argument("--end_month", type=int, required=True, help="End month.")
    parser.add_argument("--year", type=int, default=2020, help="Choosing the comparing year of resident population census or estimate.")

    args = parser.parse_args()

    # Call the main function
    create_us_state_maps(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        year=args.year
    )


