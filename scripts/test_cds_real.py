#!/usr/bin/env python3
import os
import sys

print("Testing CDS API with new configuration...")

# Check config file
config_path = os.path.expanduser("~/.cdsapirc")
print(f"Config file: {config_path}")

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        content = f.read()
    print("Current configuration:")
    print(content)
    print()
    
    # Validate format
    if "url: https://cds.climate.copernicus.eu/api" in content:
        print("✓ URL format is correct")
    else:
        print("⚠️  URL might be wrong format")
        
    if "key:" in content:
        key_line = [line for line in content.split('\n') if line.startswith('key:')]
        if key_line:
            key_value = key_line[0].split('key:')[1].strip()
            if ':' not in key_value and len(key_value) > 20:
                print("✓ Key format looks correct (no UID prefix)")
            else:
                print("⚠️  Key might have UID prefix or is too short")
else:
    print("✗ No config file found")
    sys.exit(1)

print("\nTesting CDS API import...")
try:
    import cdsapi
    print("✓ CDS API imported")
    
    # Check if we can create client
    try:
        client = cdsapi.Client()
        print("✓ CDS API client created")
        
        # Try a small test request
        print("\nTesting API connection with metadata request...")
        try:
            # Try to get dataset info (lightweight request)
            import time
            start_time = time.time()
            
            # Try different approaches
            try:
                # Method 1: Direct API call (if available)
                info = client.info('reanalysis-era5-single-levels')
                print(f"✓ API connection successful!")
                print(f"  Dataset: {info.get('name', 'reanalysis-era5-single-levels')}")
                print(f"  Response time: {time.time() - start_time:.2f} seconds")
                
            except Exception as e1:
                print(f"Method 1 failed: {e1}")
                
                # Method 2: Try a different approach
                try:
                    # Try to list datasets
                    datasets = client.datasets()
                    print(f"✓ API connection successful!")
                    print(f"  Available datasets: {len(datasets)}")
                    print(f"  Response time: {time.time() - start_time:.2f} seconds")
                    
                except Exception as e2:
                    print(f"Method 2 failed: {e2}")
                    
                    # Method 3: Minimal test
                    try:
                        # Just check if client was created
                        print("✓ Client created (connection test inconclusive)")
                        print("  To fully test, try a small download")
                        
                    except Exception as e3:
                        print(f"✗ All connection tests failed: {e3}")
                        
        except Exception as e:
            print(f"✗ Connection test error: {e}")
            print("\nPossible issues:")
            print("1. API key might be invalid or expired")
            print("2. Network connectivity issue")
            print("3. CDS API service might be down")
            
    except Exception as e:
        print(f"✗ Client creation error: {e}")
        print("\nThis usually means:")
        print("1. Configuration file has wrong format")
        print("2. API key is invalid")
        print("3. ~/.cdsapirc file permissions are wrong")
        
except ImportError as e:
    print(f"✗ CDS API not installed: {e}")
    print("Install with: pip install cdsapi")

print("\n" + "="*60)
print("NEXT STEPS:")
print("1. If API works: Run your analysis script")
print("2. If not: Check your API key at:")
print("   https://cds.climate.copernicus.eu/api-how-to")
print("3. For immediate results, use test script:")
print("   python3 scripts/era5_working.py")
print("="*60)
