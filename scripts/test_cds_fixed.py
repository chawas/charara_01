#!/usr/bin/env python3
import os

print("Testing updated CDS API configuration...")

# Check config
config_path = os.path.expanduser("~/.cdsapirc")
print(f"Config file: {config_path}")

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        print(f"Contents:\n{f.read()}")
else:
    print("⚠️  No config file found")

# Test import
try:
    import cdsapi
    print(f"✓ CDS API imported (version: {cdsapi.__version__})")
    
    # Try client
    try:
        client = cdsapi.Client()
        print("✓ Client created")
        
        # Test info request
        info = client.info('reanalysis-era5-single-levels')
        print(f"✓ API working - Dataset: {info.get('name', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Client error: {e}")
        print("This might be OK if you don't have valid API key yet")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
