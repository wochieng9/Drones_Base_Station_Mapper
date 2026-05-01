import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration - Set your API key here if using OpenRouteService
api_key = st.secrets["OPENROUTESERVICE_API_KEY"]
OPENROUTESERVICE_API_KEY = api_key
USE_OPENROUTESERVICE = False

def test_road_distance_api():
    """Test if the road distance API is working."""
    # Test coordinates: Nairobi to Mombasa
    nairobi = (-1.3009, 36.8064)
    mombasa = (-4.02796, 39.68619)
    
    print("Testing OSRM API...")
    distance = get_road_distance_osrm(nairobi, mombasa)
    
    if distance:
        print(f"✓ OSRM API working! Distance: {distance:.2f} km")
        return True
    else:
        print("✗ OSRM API failed")
        return False

def calculate_air_distance(coord1, coord2):
    """Calculate Euclidean (straight-line) distance in kilometers using Haversine formula."""
    import numpy as np
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371  # Earth's radius in kilometers
    return r * c

def get_road_distance_osrm(start_coords, end_coords, max_retries=3):
    """
    Calculate road distance using free OSRM API (no API key needed).
    
    Args:
        start_coords: tuple of (lat, lon)
        end_coords: tuple of (lat, lon)
        max_retries: number of retries on failure
    
    Returns:
        Distance in kilometers or None if failed
    """
    # OSRM expects (lon, lat) format
    url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
    
    params = {
        'overview': 'false',
        'steps': 'false'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] == 'Ok':
                distance_meters = data['routes'][0]['distance']
                return distance_meters / 1000  # Convert to km
            else:
                return None
        
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(0.5)  # Brief pause before retry
    
    return None

def get_road_distance_ors(start_coords, end_coords, api_key, max_retries=3):
    """
    Calculate road distance using OpenRouteService API (requires API key).
    
    Args:
        start_coords: tuple of (lat, lon)
        end_coords: tuple of (lat, lon)
        api_key: OpenRouteService API key
        max_retries: number of retries on failure
    
    Returns:
        Distance in kilometers or None if failed
    """
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    
    body = {
        'coordinates': [
            [start_coords[1], start_coords[0]],  # lon, lat
            [end_coords[1], end_coords[0]]
        ]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=body, headers=headers, verify=False, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            distance_meters = data['routes'][0]['summary']['distance']
            return distance_meters / 1000  # Convert to km
        
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(0.5)
    
    return None

def get_road_distance(start_coords, end_coords, api_key=None):
    """
    Calculate road distance using configured routing service.
    
    Args:
        start_coords: tuple of (lat, lon)
        end_coords: tuple of (lat, lon)
        api_key: API key for OpenRouteService (optional, only if USE_OPENROUTESERVICE=True)
    
    Returns:
        Distance in kilometers or None if failed
    """
    if USE_OPENROUTESERVICE and api_key:
        return get_road_distance_ors(start_coords, end_coords, api_key)
    else:
        return get_road_distance_osrm(start_coords, end_coords)

def create_distance_spreadsheet(optimizer, selected_bases, blood_bank_names, health_facilities, use_road_distance=True, api_key=None):
    """
    Create a spreadsheet with distances from selected bases to all health facilities.
    
    Args:
        optimizer: DroneBaseOptimizer instance
        selected_bases: list of selected base indices
        blood_bank_names: list of blood bank names
        health_facilities: numpy array of health facility coordinates
        use_road_distance: whether to calculate road distances (can be slow)
        api_key: API key for routing service (optional, only if using OpenRouteService)
    
    Returns:
        pandas DataFrame
    """
    import numpy as np
    
    rows = []
    total_calculations = len(selected_bases) * len(health_facilities)
    
    # Progress tracking
    progress_count = 0
    failed_count = 0
    
    print(f"Starting distance calculations. Road distances: {use_road_distance}")
    
    for base_idx in selected_bases:
        base_coords = optimizer.blood_banks[base_idx]
        base_name = blood_bank_names[base_idx]
        
        for facility_idx, facility_coords in enumerate(health_facilities):
            # Calculate air distance (Euclidean)
            air_distance = calculate_air_distance(base_coords, facility_coords)
            
            # Calculate road distance if requested
            road_distance = None
            if use_road_distance:
                road_distance = get_road_distance(base_coords, facility_coords, api_key)
                progress_count += 1
                
                if road_distance is None:
                    failed_count += 1
                    if failed_count <= 3:  # Only print first few errors
                        print(f"Failed to get road distance for Base {base_idx} -> Facility {facility_idx}")
                
                # Rate limiting - pause every 10 requests
                if progress_count % 10 == 0:
                    print(f"Progress: {progress_count}/{total_calculations} calculations complete")
                    time.sleep(0.2)  # Increased pause
            
            rows.append({
                'Base_Index': base_idx,
                'Base_Name': base_name,
                'Base_Latitude': base_coords[0],
                'Base_Longitude': base_coords[1],
                'Facility_Index': facility_idx,
                'Facility_Latitude': facility_coords[0],
                'Facility_Longitude': facility_coords[1],
                'Air_Distance_KM': round(air_distance, 2),
                'Road_Distance_KM': round(road_distance, 2) if road_distance is not None else 'N/A',
                'Within_Range': 'Yes' if air_distance <= optimizer.operational_radius else 'No'
            })
    
    if use_road_distance:
        print(f"Calculation complete. Failed requests: {failed_count}/{total_calculations}")
    
    df = pd.DataFrame(rows)
    
    # Add summary statistics
    if use_road_distance and df['Road_Distance_KM'].notna().any():
        # Calculate road vs air ratio where road distance is available
        valid_road = df[df['Road_Distance_KM'] != 'N/A'].copy()
        if len(valid_road) > 0:
            valid_road['Road_Distance_KM'] = pd.to_numeric(valid_road['Road_Distance_KM'])
            valid_road['Road_Air_Ratio'] = valid_road['Road_Distance_KM'] / valid_road['Air_Distance_KM']
            df = df.merge(valid_road[['Base_Index', 'Facility_Index', 'Road_Air_Ratio']], 
                         on=['Base_Index', 'Facility_Index'], how='left')
            df['Road_Air_Ratio'] = df['Road_Air_Ratio'].round(2)
    
    return df

def create_distance_spreadsheet_parallel(optimizer, selected_bases, blood_bank_names, health_facilities, max_workers=5, api_key=None):
    """
    Create distance spreadsheet with parallel processing for road distances.
    Faster but uses more network resources.
    
    Args:
        optimizer: DroneBaseOptimizer instance
        selected_bases: list of selected base indices
        blood_bank_names: list of blood bank names
        health_facilities: numpy array of health facility coordinates
        max_workers: number of parallel workers
        api_key: API key for routing service (optional)
    
    Returns:
        pandas DataFrame
    """
    import numpy as np
    
    def calculate_row(base_idx, facility_idx, base_coords, facility_coords, base_name, api_key):
        air_distance = calculate_air_distance(base_coords, facility_coords)
        road_distance = get_road_distance(base_coords, facility_coords, api_key)
        
        return {
            'Base_Index': base_idx,
            'Base_Name': base_name,
            'Base_Latitude': base_coords[0],
            'Base_Longitude': base_coords[1],
            'Facility_Index': facility_idx,
            'Facility_Latitude': facility_coords[0],
            'Facility_Longitude': facility_coords[1],
            'Air_Distance_KM': round(air_distance, 2),
            'Road_Distance_KM': round(road_distance, 2) if road_distance else 'N/A',
            'Within_Range': 'Yes' if air_distance <= optimizer.operational_radius else 'No'
        }
    
    rows = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for base_idx in selected_bases:
            base_coords = optimizer.blood_banks[base_idx]
            base_name = blood_bank_names[base_idx]
            
            for facility_idx, facility_coords in enumerate(health_facilities):
                future = executor.submit(
                    calculate_row,
                    base_idx, facility_idx, base_coords, facility_coords, base_name, api_key
                )
                futures.append(future)
        
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as e:
                print(f"Error calculating distance: {e}")
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['Base_Index', 'Facility_Index']).reset_index(drop=True)
    
    # Calculate road vs air ratio
    valid_road = df[df['Road_Distance_KM'] != 'N/A'].copy()
    if len(valid_road) > 0:
        valid_road['Road_Distance_KM'] = pd.to_numeric(valid_road['Road_Distance_KM'])
        valid_road['Road_Air_Ratio'] = valid_road['Road_Distance_KM'] / valid_road['Air_Distance_KM']
        df = df.merge(valid_road[['Base_Index', 'Facility_Index', 'Road_Air_Ratio']], 
                     on=['Base_Index', 'Facility_Index'], how='left')
        df['Road_Air_Ratio'] = df['Road_Air_Ratio'].round(2)
    
    return df
