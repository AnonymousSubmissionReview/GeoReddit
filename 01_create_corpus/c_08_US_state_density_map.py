"""
This script generates a choropleth map to visualize the density of geolocated Reddit users per 1,000 residents across U.S. states over a specified time period.

Input:
- An input folder containing:
    - p_state_counts_cumulative.csv
    - o_census.csv

Output:
- Console output:
    - Median user density per 1,000 residents
- A PNG file saved in the output folder:
    - p_YYYY-MMtoYYYY-MM_density_Maps.png
      A choropleth map displaying Reddit user density by state.

Example usage:

Visualize density from June 2005 to December 2023
python c_08_US_state_density_map.py --input_folder "C:/Users/u2288/Downloads/period" --output_folder "C:/Users/u2288/Downloads/period" --start_year 2005 --start_month 6 --end_year 2023 --end_month 12

"""
import plotly.graph_objects as go
import pandas as pd
import os
import argparse

# Generate a choropleth map showing Reddit user density per 1,000 residents by U.S. state
def plot_reddit_user_density(input_folder, output_folder, start_year, start_month, end_year, end_month,year):
    # Build the input CSV file path with p_ prefix
    file_1 = 'p_state_counts_cumulative.csv'
    file_2 = 'o_census.csv'
    file_path_1 = os.path.join(input_folder, file_1)
    file_path_2 = os.path.join(input_folder, file_2)
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)


    # Compute user density per 1,000 residents
    df = pd.merge(df1, df2, on="state", how="inner")
    df['density'] = (df[f'{end_year}_user'] / df[f'{year}'] * 1000).round().astype(int)
    median_density = df['density'].median()
    mean_density = df['density'].mean()
    print(f"Median Reddit User Density: {median_density} users per 1,000 residents")
    print(f"Mean Reddit User Density: {mean_density} users per 1,000 residents")

    # U.S. state coordinates for annotations
    state_coordinates = {
        "AL": [32.806671, -86.791130], "AK": [61.370716, -152.404419], "AZ": [33.729759, -111.431221],
        "AR": [34.969704, -92.373123], "CA": [36.116203, -119.681564], "CO": [39.059811, -105.311104],
        "CT": [41.597782, -72.755371], "DE": [39.318523, -75.507141], "FL": [27.766279, -81.686783],
        "GA": [33.040619, -83.643074], "HI": [19.7, -155.5], "ID": [44.240459, -114.478828],
        "IL": [40.349457, -88.986137], "IN": [39.849426, -86.258278], "IA": [42.011539, -93.210526],
        "KS": [38.526600, -96.726486], "KY": [37.668140, -84.670067], "LA": [31.169546, -91.867805],
        "ME": [44.693947, -69.381927], "MD": [39.063946, -76.802101], "MA": [42.230171, -71.530106],
        "MI": [43.326618, -84.536095], "MN": [45.694454, -93.900192], "MS": [32.741646, -89.678696],
        "MO": [38.456085, -92.288368], "MT": [46.921925, -110.454353], "NE": [41.125370, -98.268082],
        "NV": [38.313515, -117.055374], "NH": [43.452492, -71.563896], "NJ": [40.298904, -74.521011],
        "NM": [34.840515, -106.248482], "NY": [42.165726, -74.948051], "NC": [35.630066, -79.806419],
        "ND": [47.528912, -99.784012], "OH": [40.388783, -82.764915], "OK": [35.565342, -96.928917],
        "OR": [44.572021, -122.070938], "PA": [40.590752, -77.209755], "RI": [41.680893, -71.511780],
        "SC": [33.856892, -80.945007], "SD": [44.299782, -99.438828], "TN": [35.747845, -86.692345],
        "TX": [31.054487, -97.563461], "UT": [40.150032, -111.862434], "VT": [44.045876, -72.710686],
        "VA": [37.769337, -78.169968], "WA": [47.400902, -121.490494], "WV": [38.491226, -80.954456],
        "WI": [44.268543, -89.616508], "WY": [42.755966, -107.302490], "DC": [38.9072, -77.0369]
    }

    # Prepare state abbreviation annotations
    annotations = [
        go.Scattergeo(
            lon=[state_coordinates[state][1] for state in df['state']],
            lat=[state_coordinates[state][0] for state in df['state']],
            text=df['state'],
            mode='text',
            textfont=dict(
                family="Times New Roman",
                size=9,
                color="white"
            ),
            showlegend=False,
        )
    ]

    # Create the choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=df['state'],
        z=df['density'],
        locationmode='USA-states',
        colorscale='thermal',
        colorbar=dict(
            title="users / 10³ population",
            orientation='h',
            x=0.5,
            xanchor='center',
            thickness=15,
            len=0.6,
            y=-0.2,
        ),
    ))

    # Add annotations to the figure
    for annotation in annotations:
        fig.add_trace(annotation)

    # Configure figure layout
    fig.update_layout(
        title={
            'text': f"Reddit User Density by U.S. State ({start_year}.{start_month} - {end_year}.{end_month})",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Times New Roman'}
        },
        geo_scope='usa',
        margin={"r": 10, "t": 50, "l": 10, "b": 70},
        font=dict(
            family="Times New Roman",
            size=14
        ),
    )

    # Build the output PNG file path with p_ prefix
    output_file = os.path.join(
        output_folder,
        f"p_{start_year}-{str(start_month).zfill(2)}to{end_year}-{str(end_month).zfill(2)}_density_Maps.png"
    )
    # Save the figure as PNG
    fig.write_image(output_file, format="png", scale=3)
    print(f"Figure saved to: {output_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a choropleth map of GeoReddit user density per 1,000 residents by U.S. state."
    )
    parser.add_argument("--input_folder", required=True, help="Folder containing the input CSV file.")
    parser.add_argument("--output_folder", required=True, help="Folder to save the output image.")
    parser.add_argument("--start_year", type=int, required=True, help="Start year.")
    parser.add_argument("--start_month", type=int, required=True, help="Start month.")
    parser.add_argument("--end_year", type=int, required=True, help="End year.")
    parser.add_argument("--end_month", type=int, required=True, help="End month.")
    parser.add_argument("--year", type=int, default=2020, help="Choosing the comparing year of resident population census or estimate.")

    args = parser.parse_args()

    # Run the main function
    plot_reddit_user_density(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        year=args.year,
    )

    

