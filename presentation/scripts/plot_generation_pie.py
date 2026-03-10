#!/usr/bin/env python3
"""
Generate Power Generation Mix Pie Charts for 2015 vs 2024.

Creates pie charts showing the energy transition:
- DE: Nuclear phase-out, coal reduction, renewable expansion
- FR: Persistent nuclear dominance

Usage:
    python plot_generation_pie.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Directories
DATA_DIR = Path("/lustre/home/dthumm/CASTOR/data/unified")
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Style settings
plt.rcParams.update({
    'font.size': 10,
    'figure.dpi': 150,
})

# Color scheme for generation types
COLORS = {
    'Nuclear': '#9467bd',      # Purple
    'Lignite': '#7f7f7f',      # Gray
    'Coal': '#8c564b',         # Brown
    'Gas': '#ff7f0e',          # Orange
    'Wind': '#2ca02c',         # Green
    'Solar': '#ffdd00',        # Yellow
    'Hydro': '#1f77b4',        # Blue
    'Other': '#c7c7c7',        # Light gray
}

# Category order for consistent pie chart layout
CATEGORY_ORDER = ['Nuclear', 'Lignite', 'Coal', 'Gas', 'Wind', 'Solar', 'Hydro', 'Other']


def load_dataset(country: str) -> pd.DataFrame:
    """Load unified dataset for a country."""
    filepath = DATA_DIR / f"unified_{country}_2015_2024_clean.csv"
    return pd.read_csv(filepath, index_col=0, parse_dates=True)


def aggregate_generation(df: pd.DataFrame, year: int, country: str) -> dict:
    """
    Aggregate generation by category for a specific year.

    Returns dict of category -> mean MW.
    """
    # Filter to year
    df_year = df[df.index.year == year]

    # Define category mappings based on country
    if country == 'DE':
        categories = {
            'Nuclear': ['Nuclear_Actual Aggregated'],
            'Lignite': ['Fossil Brown coal/Lignite_Actual Aggregated'],
            'Coal': ['Fossil Hard coal_Actual Aggregated', 'Fossil Coal-derived gas_Actual Aggregated'],
            'Gas': ['Fossil Gas_Actual Aggregated', 'Fossil Oil_Actual Aggregated'],
            'Wind': ['Wind Onshore_Actual Aggregated', 'Wind Offshore_Actual Aggregated'],
            'Solar': ['Solar_Actual Aggregated'],
            'Hydro': ['Hydro Run-of-river and poundage_Actual Aggregated',
                      'Hydro Water Reservoir_Actual Aggregated',
                      'Hydro Pumped Storage_Actual Aggregated'],
            'Other': ['Biomass_Actual Aggregated', 'Waste_Actual Aggregated',
                      'Geothermal_Actual Aggregated', 'Other renewable_Actual Aggregated',
                      'Other_Actual Aggregated'],
        }
    else:  # FR
        categories = {
            'Nuclear': ['Nuclear_Actual Aggregated'],
            'Lignite': [],  # France doesn't have lignite
            'Coal': ['Fossil Hard coal_Actual Aggregated'],
            'Gas': ['Fossil Gas_Actual Aggregated', 'Fossil Oil_Actual Aggregated'],
            'Wind': ['Wind Onshore_Actual Aggregated', 'Wind Offshore_Actual Aggregated'],
            'Solar': ['Solar_Actual Aggregated'],
            'Hydro': ['Hydro Run-of-river and poundage_Actual Aggregated',
                      'Hydro Water Reservoir_Actual Aggregated',
                      'Hydro Pumped Storage_Actual Aggregated'],
            'Other': ['Biomass_Actual Aggregated', 'Waste_Actual Aggregated'],
        }

    result = {}
    for category, cols in categories.items():
        total = 0.0
        for col in cols:
            if col in df_year.columns:
                total += df_year[col].mean()
        result[category] = total

    return result


def create_pie_chart(generation: dict, title: str, output_path: Path):
    """
    Create a pie chart for generation mix.

    Args:
        generation: Dict of category -> mean MW
        title: Chart title
        output_path: Where to save the SVG
    """
    # Filter out zero/negligible categories and sort by order
    data = [(cat, generation.get(cat, 0)) for cat in CATEGORY_ORDER]
    data = [(cat, val) for cat, val in data if val > 100]  # Filter negligible (<100 MW)

    if not data:
        print(f"  Warning: No significant generation data for {title}")
        return

    labels, values = zip(*data)
    colors = [COLORS[cat] for cat in labels]

    # Calculate percentages
    total = sum(values)

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,  # We'll add a legend instead
        colors=colors,
        autopct=lambda pct: f'{pct:.0f}%' if pct > 5 else '',
        startangle=90,
        pctdistance=0.75,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    # Add legend
    legend_labels = [f'{cat} ({val/1000:.1f} GW)' for cat, val in zip(labels, values)]
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=8)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_comparison_figure(de_data: dict, fr_data: dict, output_path: Path):
    """
    Create a 2x2 comparison figure with all four pie charts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    configs = [
        (axes[0, 0], de_data[2015], 'DE 2015'),
        (axes[0, 1], de_data[2024], 'DE 2024'),
        (axes[1, 0], fr_data[2015], 'FR 2015'),
        (axes[1, 1], fr_data[2024], 'FR 2024'),
    ]

    for ax, generation, title in configs:
        # Filter and prepare data
        data = [(cat, generation.get(cat, 0)) for cat in CATEGORY_ORDER]
        data = [(cat, val) for cat, val in data if val > 100]

        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)
            continue

        labels, values = zip(*data)
        colors = [COLORS[cat] for cat in labels]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct=lambda pct: f'{pct:.0f}%' if pct > 5 else '',
            startangle=90,
            pctdistance=0.75,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
        )

        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_fontweight('bold')
            autotext.set_color('white')

        ax.set_title(title, fontsize=11, fontweight='bold')

    # Add common legend
    legend_handles = [plt.Rectangle((0,0), 1, 1, facecolor=COLORS[cat])
                      for cat in CATEGORY_ORDER if cat in COLORS]
    fig.legend(legend_handles, CATEGORY_ORDER, loc='center right',
               bbox_to_anchor=(1.12, 0.5), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("Generating Power Generation Mix Pie Charts")
    print("=" * 70)

    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")
    df_de = load_dataset('DE')
    df_fr = load_dataset('FR')
    print(f"  DE: {len(df_de)} rows")
    print(f"  FR: {len(df_fr)} rows")

    # Aggregate generation data
    print("\nAggregating generation by category...")
    de_data = {
        2015: aggregate_generation(df_de, 2015, 'DE'),
        2024: aggregate_generation(df_de, 2024, 'DE'),
    }
    fr_data = {
        2015: aggregate_generation(df_fr, 2015, 'FR'),
        2024: aggregate_generation(df_fr, 2024, 'FR'),
    }

    # Print summary
    print("\n--- DE Generation Summary (GW) ---")
    for year in [2015, 2024]:
        print(f"\n{year}:")
        total = sum(de_data[year].values())
        for cat in CATEGORY_ORDER:
            val = de_data[year].get(cat, 0)
            if val > 100:
                print(f"  {cat}: {val/1000:.2f} GW ({100*val/total:.1f}%)")

    print("\n--- FR Generation Summary (GW) ---")
    for year in [2015, 2024]:
        print(f"\n{year}:")
        total = sum(fr_data[year].values())
        for cat in CATEGORY_ORDER:
            val = fr_data[year].get(cat, 0)
            if val > 100:
                print(f"  {cat}: {val/1000:.2f} GW ({100*val/total:.1f}%)")

    # Generate individual pie charts
    print("\n--- Generating Individual Pie Charts ---")
    create_pie_chart(de_data[2015], 'DE 2015', FIGURES_DIR / 'generation_pie_de_2015.svg')
    create_pie_chart(de_data[2024], 'DE 2024', FIGURES_DIR / 'generation_pie_de_2024.svg')
    create_pie_chart(fr_data[2015], 'FR 2015', FIGURES_DIR / 'generation_pie_fr_2015.svg')
    create_pie_chart(fr_data[2024], 'FR 2024', FIGURES_DIR / 'generation_pie_fr_2024.svg')

    # Generate comparison figure
    print("\n--- Generating Comparison Figure ---")
    create_comparison_figure(de_data, fr_data, FIGURES_DIR / 'generation_pie_comparison.svg')

    print("\n" + "=" * 70)
    print(f"Pie charts saved to {FIGURES_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
