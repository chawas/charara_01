#cd /home/chawas/deployed/charara_01

# Create a no-plot version
#cat > scripts/era5_no_plots.py << 'EOF'
#!/usr/bin/env python3
"""
ERA5 Analysis WITHOUT plots
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path('/home/chawas/deployed/charara_01')
sys.path.append(str(project_root))

class ERA5Downloader:
    """Download ERA5 data for Kariba region"""
    
    def __init__(self, config_file=None):
        # ... [keep your existing __init__] ...
        pass
    # ... [keep ALL your existing methods EXCEPT create_basic_plots] ...
    
    def create_basic_plots(self, df):
        """DUMMY - Skip plotting"""
        print("\n⏭️  Skipping plot creation (debug mode)")
        print("  To enable plots, fix matplotlib backend")
        return

def main():
    print("=" * 60)
    print("ERA5 ANALYSIS - NO PLOTS VERSION")
    print("=" * 60)
    
    # Initialize downloader
    downloader = ERA5Downloader()
    
    print("\nWhat would you like to do?")
    print("1. Download ERA5 data (requires CDS API key)")
    print("2. Process existing ERA5 files")
    print("3. Quick test (no download)")
    
    choice = "2"  # Auto-choose option 2
    print(f"\nAuto-selected: {choice}")
    
    if choice == '2':
        print("\nProcessing existing ERA5 files...")
        
        ds = downloader.process_downloaded_data()
        
        if ds is not None:
            df = downloader.extract_charara_data(ds)
            downloader.create_basic_plots(df)  # This will skip
            
            print("\n" + "=" * 60)
            print("✅ PROCESSING COMPLETE!")
            print("=" * 60)
            print(f"\nOutputs created:")
            print(f"  • data/era5_processed/era5_kariba_combined.nc")
            print(f"  • outputs/charara_era5_timeseries.csv")
            print(f"  • outputs/charara_statistics.txt")
            print(f"\nData covers: {df.index[0]} to {df.index[-1]}")
            print(f"Total records: {len(df):,}")
    
    print("\n✅ Script completed successfully!")

if __name__ == "__main__":
    main()
EOF

# Run the no-plots version
#python scripts/era5_no_plots.py