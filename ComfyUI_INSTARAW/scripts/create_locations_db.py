import csv
import json
import argparse
from collections import defaultdict
import os # Import os for directory creation

def create_locations_database(csv_path, region_map_path, output_path, min_population, bbox_offset):
    """
    Parses a world cities CSV and creates a hierarchical JSON database for GPS synthesis.
    """
    print("Loading country to region mapping...")
    with open(region_map_path, 'r', encoding='utf-8') as f:
        region_data = json.load(f)

    # Create a reverse map for quick lookup: country -> region
    country_to_region = {}
    for region, countries in region_data.items():
        for country in countries:
            country_to_region[country] = region

    print(f"Processing cities from '{csv_path}'...")
    print(f"Filtering for cities with population > {min_population:,}")

    locations = defaultdict(lambda: defaultdict(list))
    
    city_count = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                population = float(row.get('population', 0))
                if population < min_population:
                    continue

                country = row['country']
                region = country_to_region.get(country)
                if not region:
                    continue

                lat = float(row['lat'])
                lon = float(row['lng'])

                city_data = {
                    "city": row['city'],
                    "lat": [lat - bbox_offset, lat + bbox_offset],
                    "lon": [lon - bbox_offset, lon + bbox_offset]
                }
                
                locations[region][country].append(city_data)
                city_count += 1

            except (ValueError, TypeError):
                continue
    
    print(f"Processed {city_count} cities.")
    
    # --- MODIFIED: Ensure the output directory exists ---
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        print(f"Creating directory: '{output_dir}'")
        os.makedirs(output_dir)
        
    print(f"Saving database to '{output_path}'...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(locations, f, indent=2)

    print("\n✅ Success! Location database created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a location database from worldcities.csv.")
    parser.add_argument("--csv-path", default="worldcities.csv", help="Path to the worldcities.csv file.")
    parser.add_argument("--region-map-path", default="country_to_region.json", help="Path to the country_to_region.json helper file.")
    # --- MODIFIED: New default output path ---
    parser.add_argument("--output-path", default="../modules/location_data/locations.json", help="Path for the output locations.json file.")
    parser.add_argument("--min-population", type=int, default=1000000, help="Minimum population to include a city.")
    parser.add_argument("--bbox-offset", type=float, default=0.2, help="Degrees to add/subtract from lat/lon to create a bounding box.")
    
    args = parser.parse_args()
    create_locations_database(args.csv_path, args.region_map_path, args.output_path, args.min_population, args.bbox_offset)