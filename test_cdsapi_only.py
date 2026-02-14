#!/usr/bin/env python3
"""
Clean CDS API test without ecmwf-datastores
"""

import os
import sys
import time

print("Testing CDS API (cdsapi library only)...")

# Check which libraries are installed
print("\n1. Checking installed libraries...")
try:
    import cdsapi
    print("   ✓ cdsapi installed")
except ImportError:
    print("   ✗ cdsapi not installed")
    print("   Install with: pip install cdsapi")
    sys.exit(1)

try:
    import ecmwf.datastores
    print("   ⚠️  ecmwf.datastores also installed (might cause conflicts)")
except ImportError:
    print("   ✓ ecmwf.datastores not installed (good)")

# Check config
print("\n2. Checking configuration...")
config_path = os.path.expanduser("~/.cdsapirc")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        print(f"   Config file exists:\n{f.read()}")
else:
    print("   ✗ No ~/.cdsapirc file")
    sys.exit(1)

# Test basic connection
print("\n3. Testing connection...")
try:
    # Create client with timeout
    client = cdsapi.Client(timeout=30, quiet=True)
    print("   ✓ Client created")
    
    # Try a VERY small test request
    print("\n4. Testing with minimal request...")
    
    test_request = {
        'product_type': 'reanalysis',
        'variable': '10m_u_component_of_wind',
        'year': '2020',
        'month': '01',
        'day': '01',
        'time': '00:00',
        'area': [-16.6, 28.8, -16.7, 28.9],  # Tiny area
        'format': 'netcdf',
    }
    
    print("   Request parameters:")
    for key, value in test_request.items():
        print(f"     {key}: {value}")
    
    # Create output file
    import tempfile
    import shutil
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    output_file = os.path.join(temp_dir, 'test.nc')
    
    print(f"\n   Output: {output_file}")
    print("   Submitting request (may take a minute)...")
    
    start_time = time.time()
    
    # Submit request
    result = client.retrieve('reanalysis-era5-single-levels', test_request, output_file)
    
    elapsed = time.time() - start_time
    
    # Check if file was created
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"\n   ✅ SUCCESS! File downloaded in {elapsed:.1f} seconds")
        print(f"   Size: {os.path.getsize(output_file) / 1024:.1f} KB")
        
        # Try to read the file
        try:
            import xarray as xr
            ds = xr.open_dataset(output_file)
            print(f"   Variables: {list(ds.data_vars)}")
            print(f"   Dimensions: {dict(ds.dims)}")
            ds.close()
        except:
            print("   File created but could not read with xarray")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
    else:
        print(f"\n   ⚠️  Request submitted but no file created yet")
        print("   Check: https://cds.climate.copernicus.eu/cdsapp#!/yourrequests")
        print(f"   Request took {elapsed:.1f} seconds")
        
except Exception as e:
    print(f"\n   ❌ Error: {type(e).__name__}: {str(e)[:200]}")
    
    # Provide specific solutions based on error
    if "404" in str(e):
        print("\n   Solution: Update your ~/.cdsapirc file:")
        print("     url: https://cds.climate.copernicus.eu/api/v2")
        print("     key: YOUR_KEY_WITHOUT_UID_PREFIX")
    elif "timeout" in str(e).lower():
        print("\n   Solution: Network timeout - try again later")
    elif "resolve" in str(e).lower():
        print("\n   Solution: DNS issue - check network connection")
    else:
        print("\n   Solution: Check API key and configuration")

print("\n" + "="*60)
print("SUMMARY:")
print("Your DNS is working (resolves to 136.156.139.54)")
print("Ping fails (firewall may block ICMP)")
print("HTTP works (got 200 response)")
print("\nNext: Try the clean CDS API test above")
print("="*60)
