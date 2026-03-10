"""
Central country configuration for the electricity forecasting pipeline.

Adding a new country requires only adding an entry to COUNTRY_REGISTRY below.
No other code changes are needed.
"""

from typing import List, Tuple, Optional
from datetime import date


COUNTRY_REGISTRY = {
    'DE': {
        'name': 'Germany',
        'entsoe_zones': {
            'current': 'DE_LU',
            'historical': [
                {'code': 'DE_AT_LU', 'until': '2018-10-01'},
            ],
        },
        'weather': {
            'cities': [
                {'name': 'Berlin', 'lat': 52.52, 'lon': 13.405},
                {'name': 'Munich', 'lat': 48.137, 'lon': 11.575},
                {'name': 'Frankfurt', 'lat': 50.110, 'lon': 8.682},
                {'name': 'Hamburg', 'lat': 53.551, 'lon': 9.993},
                {'name': 'Cologne', 'lat': 50.937, 'lon': 6.960},
            ],
            'grid_bbox': (5.99, 47.30, 15.02, 54.98),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'DE',
        'gas_storage_code': 'DE',
    },
    'FR': {
        'name': 'France',
        'entsoe_zones': {
            'current': 'FR',
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Paris', 'lat': 48.857, 'lon': 2.352},
                {'name': 'Lyon', 'lat': 45.764, 'lon': 4.835},
                {'name': 'Marseille', 'lat': 43.297, 'lon': 5.381},
                {'name': 'Toulouse', 'lat': 43.605, 'lon': 1.444},
                {'name': 'Lille', 'lat': 50.630, 'lon': 3.057},
            ],
            'grid_bbox': (-4.79, 41.36, 9.56, 51.15),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'FR',
        'gas_storage_code': 'FR',
    },
    'NL': {
        'name': 'Netherlands',
        'entsoe_zones': {
            'current': 'NL',
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Amsterdam', 'lat': 52.370, 'lon': 4.895},
                {'name': 'Rotterdam', 'lat': 51.924, 'lon': 4.478},
                {'name': 'Utrecht', 'lat': 52.091, 'lon': 5.122},
                {'name': 'Groningen', 'lat': 53.219, 'lon': 6.567},
                {'name': 'Maastricht', 'lat': 50.851, 'lon': 5.691},
            ],
            'grid_bbox': (3.37, 50.75, 7.22, 53.48),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'NL',
        'gas_storage_code': 'NL',
    },
    'BE': {
        'name': 'Belgium',
        'entsoe_zones': {
            'current': 'BE',
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Brussels', 'lat': 50.850, 'lon': 4.352},
                {'name': 'Antwerp', 'lat': 51.220, 'lon': 4.402},
                {'name': 'Ghent', 'lat': 51.054, 'lon': 3.721},
                {'name': 'Liege', 'lat': 50.633, 'lon': 5.568},
                {'name': 'Charleroi', 'lat': 50.411, 'lon': 4.445},
            ],
            'grid_bbox': (2.38, 49.50, 6.40, 51.51),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'BE',
        'gas_storage_code': 'BE',
    },
    'AT': {
        'name': 'Austria',
        'entsoe_zones': {
            'current': 'AT',
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Vienna', 'lat': 48.208, 'lon': 16.373},
                {'name': 'Graz', 'lat': 47.070, 'lon': 15.439},
                {'name': 'Linz', 'lat': 48.306, 'lon': 14.286},
                {'name': 'Salzburg', 'lat': 47.811, 'lon': 13.055},
                {'name': 'Innsbruck', 'lat': 47.269, 'lon': 11.394},
            ],
            'grid_bbox': (9.53, 46.37, 17.16, 49.02),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'AT',
        'gas_storage_code': 'AT',
    },
    'IT': {
        'name': 'Italy',
        'entsoe_zones': {
            'current': 'IT_NORD',
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Rome', 'lat': 41.902, 'lon': 12.496},
                {'name': 'Milan', 'lat': 45.464, 'lon': 9.190},
                {'name': 'Naples', 'lat': 40.852, 'lon': 14.268},
                {'name': 'Turin', 'lat': 45.070, 'lon': 7.687},
                {'name': 'Florence', 'lat': 43.770, 'lon': 11.249},
            ],
            'grid_bbox': (6.63, 36.62, 18.52, 47.12),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'IT',
        'gas_storage_code': 'IT',
    },
    'ES': {
        'name': 'Spain',
        'entsoe_zones': {
            'current': 'ES',
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Madrid', 'lat': 40.417, 'lon': -3.704},
                {'name': 'Barcelona', 'lat': 41.389, 'lon': 2.169},
                {'name': 'Valencia', 'lat': 39.470, 'lon': -0.376},
                {'name': 'Seville', 'lat': 37.389, 'lon': -5.984},
                {'name': 'Bilbao', 'lat': 43.263, 'lon': -2.925},
            ],
            'grid_bbox': (-9.30, 35.95, 4.33, 43.80),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'ES',
        'gas_storage_code': 'ES',
    },
    'PL': {
        'name': 'Poland',
        'entsoe_zones': {
            'current': 'PL',
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Warsaw', 'lat': 52.230, 'lon': 21.012},
                {'name': 'Krakow', 'lat': 50.050, 'lon': 19.945},
                {'name': 'Wroclaw', 'lat': 51.110, 'lon': 17.039},
                {'name': 'Poznan', 'lat': 52.407, 'lon': 16.934},
                {'name': 'Gdansk', 'lat': 54.372, 'lon': 18.647},
            ],
            'grid_bbox': (14.12, 49.00, 24.15, 54.84),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'PL',
        'gas_storage_code': 'PL',
    },
    'DK': {
        'name': 'Denmark',
        'entsoe_zones': {
            'current': ['DK_1', 'DK_2'],  # Two zones, download both and aggregate
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Copenhagen', 'lat': 55.676, 'lon': 12.569},
                {'name': 'Aarhus', 'lat': 56.163, 'lon': 10.214},
                {'name': 'Odense', 'lat': 55.396, 'lon': 10.389},
                {'name': 'Aalborg', 'lat': 57.048, 'lon': 9.934},
                {'name': 'Esbjerg', 'lat': 55.470, 'lon': 8.452},
            ],
            'grid_bbox': (8.07, 54.56, 15.25, 57.75),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'DK',
        'gas_storage_code': 'DK',
    },
    'SE': {
        'name': 'Sweden',
        'entsoe_zones': {
            'current': ['SE_1', 'SE_2', 'SE_3', 'SE_4'],  # Four zones, aggregate
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Stockholm', 'lat': 59.329, 'lon': 18.069},
                {'name': 'Gothenburg', 'lat': 57.709, 'lon': 11.975},
                {'name': 'Malmo', 'lat': 55.605, 'lon': 13.000},
                {'name': 'Uppsala', 'lat': 59.859, 'lon': 17.639},
                {'name': 'Vasteras', 'lat': 59.611, 'lon': 16.544},
            ],
            'grid_bbox': (10.56, 55.34, 24.18, 69.06),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'SE',
        'gas_storage_code': None,  # Sweden has no significant gas storage
    },
    'HU': {
        'name': 'Hungary',
        'entsoe_zones': {
            'current': 'HU',
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Budapest', 'lat': 47.498, 'lon': 19.041},
                {'name': 'Debrecen', 'lat': 47.532, 'lon': 21.626},
                {'name': 'Szeged', 'lat': 46.253, 'lon': 20.141},
                {'name': 'Miskolc', 'lat': 48.103, 'lon': 20.778},
                {'name': 'Pecs', 'lat': 46.072, 'lon': 18.233},
            ],
            'grid_bbox': (16.11, 45.74, 22.90, 48.63),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'HU',
        'gas_storage_code': 'HU',
    },
    'CZ': {
        'name': 'Czech Republic',
        'entsoe_zones': {
            'current': 'CZ',
            'historical': [],
        },
        'weather': {
            'cities': [
                {'name': 'Prague', 'lat': 50.076, 'lon': 14.438},
                {'name': 'Brno', 'lat': 49.195, 'lon': 16.608},
                {'name': 'Ostrava', 'lat': 49.836, 'lon': 18.283},
                {'name': 'Plzen', 'lat': 49.748, 'lon': 13.378},
                {'name': 'Usti nad Labem', 'lat': 50.661, 'lon': 14.032},
            ],
            'grid_bbox': (12.09, 48.55, 18.86, 51.06),
            'grid_resolution': 0.5,
        },
        'holidays_iso': 'CZ',
        'gas_storage_code': 'CZ',
    },
}


# Cross-border interconnections: set of frozensets for undirected pairs.
# Only includes pairs with actual ENTSO-E tracked transmission capacity.
INTERCONNECTIONS = {
    frozenset({'DE', 'FR'}),
    frozenset({'DE', 'NL'}),
    frozenset({'DE', 'BE'}),
    frozenset({'DE', 'AT'}),
    frozenset({'DE', 'PL'}),
    frozenset({'DE', 'DK'}),
    frozenset({'DE', 'SE'}),
    frozenset({'DE', 'CZ'}),
    frozenset({'FR', 'BE'}),
    frozenset({'FR', 'IT'}),
    frozenset({'FR', 'ES'}),
    frozenset({'NL', 'BE'}),
    frozenset({'NL', 'DK'}),
    frozenset({'AT', 'IT'}),
    frozenset({'AT', 'HU'}),
    frozenset({'AT', 'CZ'}),
    frozenset({'PL', 'CZ'}),
    frozenset({'PL', 'SE'}),
    frozenset({'DK', 'SE'}),
}


# Zone-level interconnections for multi-zone countries (used for flow downloads).
# Maps country pairs to the specific bidding zone pairs used for cross-border flows.
ZONE_INTERCONNECTIONS = {
    ('DE', 'DK'): [('DE_LU', 'DK_1')],
    ('DE', 'SE'): [('DE_LU', 'SE_4')],
    ('DK', 'SE'): [('DK_1', 'SE_3'), ('DK_2', 'SE_4')],
    ('DK', 'NL'): [('DK_1', 'NL')],
    ('PL', 'SE'): [('PL', 'SE_4')],
    ('FR', 'IT'): [('FR', 'IT_NORD')],
    ('AT', 'IT'): [('AT', 'IT_NORD')],
}


def get_entsoe_zone(country: str, query_date: str = None) -> str:
    """Return the correct ENTSO-E bidding zone code for a country.

    For multi-zone countries (DK, SE), returns the first/primary zone.
    Use get_entsoe_zones() to get all zones.

    Args:
        country: Country code (e.g., 'DE', 'FR')
        query_date: Date string 'YYYY-MM-DD' to handle historical zone changes.
                    If None, returns the current zone.
    """
    if country not in COUNTRY_REGISTRY:
        return country  # Passthrough for unknown codes

    zone_config = COUNTRY_REGISTRY[country]['entsoe_zones']

    # Check historical zones
    if query_date and zone_config.get('historical'):
        for hist in zone_config['historical']:
            if query_date < hist['until']:
                return hist['code']

    current = zone_config['current']
    if isinstance(current, list):
        return current[0]  # Primary zone for multi-zone countries
    return current


def get_entsoe_zones(country: str) -> List[str]:
    """Return all ENTSO-E bidding zone codes for a country.

    For single-zone countries, returns a list with one element.
    For multi-zone countries (DK, SE), returns all sub-zones.
    """
    if country not in COUNTRY_REGISTRY:
        return [country]

    current = COUNTRY_REGISTRY[country]['entsoe_zones']['current']
    if isinstance(current, list):
        return current
    return [current]


def is_multi_zone(country: str) -> bool:
    """Check if a country has multiple bidding zones."""
    if country not in COUNTRY_REGISTRY:
        return False
    return isinstance(COUNTRY_REGISTRY[country]['entsoe_zones']['current'], list)


def get_neighbors(country: str) -> List[str]:
    """Return list of countries interconnected with the given country."""
    neighbors = []
    for pair in INTERCONNECTIONS:
        if country in pair:
            other = next(iter(pair - {country}))
            neighbors.append(other)
    return sorted(neighbors)


def get_all_pairs() -> List[Tuple[str, str]]:
    """Return all interconnected country pairs as sorted tuples."""
    return sorted(tuple(sorted(pair)) for pair in INTERCONNECTIONS)


def get_country_pairs_for(country: str) -> List[Tuple[str, str]]:
    """Return all interconnected pairs involving a given country."""
    pairs = []
    for pair in INTERCONNECTIONS:
        if country in pair:
            pairs.append(tuple(sorted(pair)))
    return sorted(pairs)


def get_flow_zone_pairs(country_a: str, country_b: str) -> List[Tuple[str, str]]:
    """Return the specific bidding zone pairs for cross-border flow downloads.

    For simple cases (both single-zone), returns [(zone_a, zone_b)].
    For multi-zone cases, uses ZONE_INTERCONNECTIONS lookup.
    """
    key = tuple(sorted([country_a, country_b]))
    if key in ZONE_INTERCONNECTIONS:
        return ZONE_INTERCONNECTIONS[key]

    # Default: use primary zones
    zone_a = get_entsoe_zone(country_a)
    zone_b = get_entsoe_zone(country_b)
    return [(zone_a, zone_b)]


def has_gas_storage(country: str) -> bool:
    """Check if a country has gas storage tracked by AGSI+."""
    if country not in COUNTRY_REGISTRY:
        return False
    return COUNTRY_REGISTRY[country].get('gas_storage_code') is not None


def get_registered_countries() -> List[str]:
    """Return all registered country codes."""
    return sorted(COUNTRY_REGISTRY.keys())
