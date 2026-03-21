"""
Charlotte Metro WFH Analysis - 2024 UPDATE
Uses Census API directly (latest 2024 data)

Installation:
uv add requests pandas matplotlib seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests

# =============================================================================
# SETUP & CONSTANTS
# =============================================================================

CHARLOTTE_METRO_COUNTIES = [
    'Mecklenburg', 'Union', 'Cabarrus', 'Gaston', 'Iredell', 
    'York (SC)', 'Lancaster (SC)'
]

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_real_county_data(state_fips='37', include_sc=True, metro_only=True):
    """
    Fetch real ACS data from Census API
    """
    base_url = "https://api.census.gov/data"
    
    variables = [
        'NAME',
        'B08301_021E',    # WFH workers
        'B08301_001E',    # Total workers
        'B19013_001E',    # Median income
        'B28002_004E',    # Has broadband
        'B28002_001E',    # Total households
    ]
    vars_str = ','.join(variables)
    
    all_data = []
    states_to_fetch = [state_fips]
    if include_sc:
        states_to_fetch.append('45')
    
    for state in states_to_fetch:
        state_name = 'North Carolina' if state == '37' else 'South Carolina'
        print(f"Fetching data for {state_name}...")
        
        # 2019 baseline
        url_2019 = f"{base_url}/2019/acs/acs5"
        # 2024 current (latest available as of Jan 2026)
        url_2024 = f"{base_url}/2024/acs/acs5"
        
        params = {'get': vars_str, 'for': 'county:*', 'in': f'state:{state}'}
        
        resp_2019 = requests.get(url_2019, params=params)
        resp_2024 = requests.get(url_2024, params=params)
        
        if resp_2019.status_code == 200 and resp_2024.status_code == 200:
            data_2019 = resp_2019.json()
            data_2024 = resp_2024.json()
            
            df_2019 = pd.DataFrame(data_2019[1:], columns=data_2019[0])
            df_2024 = pd.DataFrame(data_2024[1:], columns=data_2024[0])
            
            df_2019['GEOID'] = df_2019['state'] + df_2019['county']
            df_2024['GEOID'] = df_2024['state'] + df_2024['county']
            
            df_2019 = df_2019.rename(columns={
                'B08301_021E': 'WFH_Workers_2019',
                'B08301_001E': 'Total_Workers_2019',
            })
            
            df_2024 = df_2024.rename(columns={
                'B08301_021E': 'WFH_Workers_2024',
                'B08301_001E': 'Total_Workers_2024',
                'B19013_001E': 'Median_Income',
                'B28002_004E': 'Broadband_HH',
                'B28002_001E': 'Total_HH',
            })
            
            df_merged = pd.merge(
                df_2019[['GEOID', 'WFH_Workers_2019', 'Total_Workers_2019']],
                df_2024[['GEOID', 'NAME', 'WFH_Workers_2024', 'Total_Workers_2024', 
                        'Median_Income', 'Broadband_HH', 'Total_HH']],
                on='GEOID'
            )
            all_data.append(df_merged)
    
    if not all_data:
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Clean county names
    df['County'] = df['NAME'].str.replace(', North Carolina', '').str.replace(', South Carolina', ' (SC)').str.replace(' County', '')
    
    # Filter to Charlotte Metro if requested
    if metro_only:
        df = df[df['County'].isin(CHARLOTTE_METRO_COUNTIES)]
    
    # Convert to numeric
    numeric_cols = ['WFH_Workers_2019', 'Total_Workers_2019', 'WFH_Workers_2024', 
                   'Total_Workers_2024', 'Median_Income', 'Broadband_HH', 'Total_HH']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Broadband_Pct'] = (df['Broadband_HH'] / df['Total_HH'] * 100).round(1)
    
    # Hardcoded land areas for Charlotte metro counties
    land_areas = {'Mecklenburg': 546, 'Union': 640, 'Cabarrus': 364, 'Gaston': 364, 'Iredell': 597, 'York (SC)': 683, 'Lancaster (SC)': 555}
    df['Land_Area_SqMi'] = df['County'].map(land_areas).fillna(500)
    
    print(f"✓ Processed {len(df)} counties")
    return df

def calculate_metrics(df):
    """Add derived columns"""
    df['WFH_Rate_2019'] = (df['WFH_Workers_2019'] / df['Total_Workers_2019'] * 100).round(1)
    df['WFH_Rate_2024'] = (df['WFH_Workers_2024'] / df['Total_Workers_2024'] * 100).round(1)
    df['WFH_Growth_Pct'] = ((df['WFH_Rate_2024'] - df['WFH_Rate_2019']) / df['WFH_Rate_2019'] * 100).round(1)
    df['Broadband_Gap'] = 100 - df['Broadband_Pct']
    df['Market_Value_Score'] = (df['WFH_Workers_2024'] * df['Median_Income'] * df['Broadband_Gap'] / 1e9).round(2)
    return df

# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def chart_1_wfh_growth(df):
    """Bar chart: 2019 vs 2024 WFH rates by county"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['WFH_Rate_2019'], width, label='2019', color='steelblue')
    ax.bar(x + width/2, df['WFH_Rate_2024'], width, label='2024', color='coral')
    
    ax.set_ylabel('WFH Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('WFH Growth: Charlotte Metro (2019 vs 2024)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['County'], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("CHARLOTTE METRO WFH ANALYSIS - 2024 DATA")
    df_counties = fetch_real_county_data(metro_only=True)
    if df_counties is not None:
        df_counties = calculate_metrics(df_counties)
        fig1 = chart_1_wfh_growth(df_counties)
        plt.savefig('chart1_wfh_growth_2024.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nTop Counties by 2024 WFH Rate:")
        print(df_counties[['County', 'WFH_Rate_2024', 'WFH_Growth_Pct']].to_string(index=False))