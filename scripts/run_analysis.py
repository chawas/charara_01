#!/usr/bin/env python3
"""
run_kariba.py - Python wrapper for Kariba Solar Project
Handles Conda environment and runs ERA5 analysis.
Run with: python run_kariba.py
"""

import sys
import os
import subprocess
import signal
from pathlib import Path
import argparse


#!/usr/bin/env python3
import sys
print(f"DEBUG - Python executable: {sys.executable}")
print(f"DEBUG - Python version: {sys.version}")
print(f"DEBUG - sys.path: {sys.path}")

# Then your existing imports...
import matplotlib.pyplot as plt
# ...
class KaribaProjectRunner:
    """Runner for Kariba Solar Project ERA5 analysis"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.main_script = self.project_root / "era5_analysis_05.py"
        self.conda_env = "deploy_env"
        
    def check_conda_available(self):
        """Check if Conda is installed and available"""
        try:
            result = subprocess.run(
                ["conda", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Conda found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Conda not found or not in PATH")
            print("Please install Conda or add it to your PATH")
            return False
    
    def check_environment_exists(self):
        """Check if the Conda environment exists"""
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            
            envs = [line.split()[0] for line in result.stdout.split('\n') 
                   if line and not line.startswith('#')]
            
            if self.conda_env in envs:
                print(f"✓ Conda environment '{self.conda_env}' exists")
                return True
            else:
                print(f"✗ Conda environment '{self.conda_env}' not found")
                print("Available environments:")
                for env in envs:
                    print(f"  - {env}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Error checking Conda environments: {e}")
            return False
    
    def get_python_path(self):
        """Get Python path from Conda environment"""
        try:
            result = subprocess.run(
                ["conda", "run", "-n", self.conda_env, "which", "python"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            # Fallback to default Python in environment
            conda_prefix = subprocess.run(
                ["conda", "info", "--base"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            return os.path.join(conda_prefix, "envs", self.conda_env, "bin", "python")
    
    def check_dependencies(self):
        """Check if required packages are installed in the environment"""
        check_script = f"""
import sys
try:
    import xarray
    import netCDF4
    import h5netcdf
    import numpy
    import pandas
    print("SUCCESS: All dependencies found")
except ImportError as e:
    print(f"MISSING: {{e.name}}")
    sys.exit(1)
"""
        
        try:
            result = subprocess.run(
                ["conda", "run", "-n", self.conda_env, "python", "-c", check_script],
                capture_output=True,
                text=True,
                check=True
            )
            print("✓ All dependencies satisfied")
            return True
        except subprocess.CalledProcessError as e:
            print("✗ Missing dependencies detected")
            print(f"Error: {e.stderr}")
            return False
    
    def install_dependencies(self):
        """Install missing dependencies"""
        print("Installing required packages...")
        
        packages = ["xarray", "netCDF4", "h5netcdf", "numpy", "pandas", "scipy", "cfgrib", "cdsapi"]
        
        try:
            # Try conda first
            print("Trying conda install...")
            subprocess.run(
                ["conda", "install", "-n", self.conda_env] + packages + ["-y", "-c", "conda-forge"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("✓ Packages installed via conda")
            return True
        except subprocess.CalledProcessError:
            # Fallback to pip
            try:
                print("Trying pip install...")
                subprocess.run(
                    ["conda", "run", "-n", self.conda_env, "pip", "install"] + packages,
                    check=True
                )
                print("✓ Packages installed via pip")
                return True
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install packages: {e}")
                return False
    
    def run_analysis(self, args=None):
        """Run the main ERA5 analysis script"""
        print(f"\n{'='*60}")
        print("Running Kariba Solar ERA5 Analysis")
        print(f"{'='*60}")
        
        if not self.main_script.exists():
            print(f"✗ Main script not found: {self.main_script}")
            return False
        
        python_path = self.get_python_path()
        
        # Build command
        cmd = [python_path, str(self.main_script)]
        if args and args != ["--"]:
            cmd.extend(args)
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Working directory: {self.project_root}")
        print(f"{'-'*60}")
        
        try:
            # Run the analysis
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )
            
            # Handle Ctrl+C gracefully
            def signal_handler(sig, frame):
                print("\nReceived interrupt signal, terminating...")
                process.terminate()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code == 0:
                print(f"\n{'='*60}")
                print("✓ Analysis completed successfully!")
                print(f"{'='*60}")
                return True
            else:
                print(f"\n{'='*60}")
                print(f"✗ Analysis failed with exit code: {return_code}")
                print(f"{'='*60}")
                return False
                
        except Exception as e:
            print(f"✗ Error running analysis: {e}")
            return False
    
    def create_conda_environment(self):
        """Create the Conda environment if it doesn't exist"""
        print(f"Creating Conda environment '{self.conda_env}'...")
        
        try:
            subprocess.run([
                "conda", "create", "-n", self.conda_env,
                "python=3.9", "xarray", "netcdf4", "h5netcdf",
                "numpy", "scipy", "pandas", "matplotlib",
                "cfgrib", "cdsapi", "-y", "-c", "conda-forge"
            ], check=True)
            print(f"✓ Environment '{self.conda_env}' created successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create environment: {e}")
            return False
    
    def run(self, args=None):
        """Main runner method"""
        print(f"{'='*60}")
        print("Kariba Solar Project Runner")
        print(f"{'='*60}")
        
        # Step 1: Check Conda
        if not self.check_conda_available():
            return False
        
        # Step 2: Check/Create environment
        if not self.check_environment_exists():
            create = input(f"\nCreate environment '{self.conda_env}'? (y/n): ").strip().lower()
            if create == 'y':
                if not self.create_conda_environment():
                    return False
            else:
                return False
        
        # Step 3: Check dependencies
        if not self.check_dependencies():
            install = input("\nInstall missing dependencies? (y/n): ").strip().lower()
            if install == 'y':
                if not self.install_dependencies():
                    return False
            else:
                return False
        
        # Step 4: Run analysis
        return self.run_analysis(args)

def main():
    parser = argparse.ArgumentParser(description="Kariba Solar Project ERA5 Analysis Runner")
    parser.add_argument(
        "args", 
        nargs=argparse.REMAINDER,
        help="Arguments to pass to era5_analysis_05.py"
    )
    
    parsed_args = parser.parse_args()
    
    # Filter out '--' if present
    cmd_args = [arg for arg in parsed_args.args if arg != '--']
    
    runner = KaribaProjectRunner()
    success = runner.run(cmd_args)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()