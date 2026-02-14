#!/usr/bin/env python3
"""
project_setup.py - Complete Kariba Wind Analysis Project Setup
"""
import os
import sys
import json
import yaml
import shutil
from pathlib import Path
import subprocess
from datetime import datetime

class KaribaProjectSetup:
    """Setup complete Kariba wind analysis project"""
    
    def __init__(self, project_dir=None):
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {
            'project': {
                'name': 'Kariba Wind Analysis',
                'description': '30-year wind regime analysis of Lake Kariba',
                'version': '1.0.0',
                'author': 'Webster Chawaguta',
                'email': 'wchawaguta@example.com',
                'start_year': 1990,
                'end_year': 2020
            },
            'lake': {
                'name': 'Kariba',
                'charara_point': [-16.53, 28.83],
                'dam_point': [-16.52, 28.76],
                'avg_depth': 29.0,
                'max_depth': 97.0,
                'coordinates': {
                    'north': -15.75,
                    'south': -17.25,
                    'west': 27.25,
                    'east': 29.00
                }
            },
            'paths': {
                'python': '/home/chawas/deployed/deployed_env/bin/python3',
                'wrf_dir': '/path/to/WRF',
                'wps_dir': '/path/to/WPS',
                'geog_data': '/path/to/WPS_GEOG'
            }
        }
    
    def create_project_structure(self):
        """Create complete project directory structure"""
        
        print("=" * 60)
        print("CREATING PROJECT STRUCTURE")
        print("=" * 60)
        
        directories = [
            # Main analysis modules
            '00_era5_analysis',
            '01_lake_preparation',
            '02_wrf_configuration',
            '03_simulation_automation',
            '04_post_processing',
            '05_validation_analysis',
            '06_wind_regime_analysis',
            '07_visualization',
            
            # Support directories
            'config',
            'scripts',
            'notebooks',
            'docs',
            'tests',
            
            # Data directories
            'data/era5/raw',
            'data/era5/processed',
            'data/observations',
            'data/bathymetry',
            'data/wrf_inputs',
            'data/wrf_outputs',
            
            # Output directories
            'outputs/figures',
            'outputs/reports',
            'outputs/timeseries',
            'outputs/validation',
            'outputs/trends',
            
            # Temporary and working directories
            'temp',
            'logs',
            'cache'
        ]
        
        for directory in directories:
            dir_path = self.project_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory}")
        
        # Create __init__.py files
        for module in ['00_era5_analysis', '01_lake_preparation', '02_wrf_configuration',
                      '03_simulation_automation', '04_post_processing', '05_validation_analysis',
                      '06_wind_regime_analysis', '07_visualization', 'scripts', 'tests']:
            init_file = self.project_dir / module / '__init__.py'
            init_file.touch()
        
        print(f"\nProject structure created at: {self.project_dir}")
        return True
    
    def create_vscode_config(self):
        """Create VSCode configuration"""
        
        print("\n" + "=" * 60)
        print("CREATING VSCODE CONFIGURATION")
        print("=" * 60)
        
        vscode_dir = self.project_dir / '.vscode'
        vscode_dir.mkdir(exist_ok=True)
        
        # Get Python path from config
        python_path = self.config['paths']['python']
        
        # settings.json
        settings = {
            "python.defaultInterpreterPath": python_path,
            "python.terminal.activateEnvironment": True,
            "python.terminal.activateEnvInCurrentTerminal": True,
            "python.linting.enabled": True,
            "python.linting.pylintEnabled": True,
            "python.linting.flake8Enabled": True,
            "python.formatting.provider": "black",
            "python.formatting.blackArgs": ["--line-length", "88"],
            "python.analysis.typeCheckingMode": "basic",
            "python.analysis.autoImportCompletions": True,
            "python.analysis.extraPaths": [
                "./00_era5_analysis",
                "./01_lake_preparation",
                "./02_wrf_configuration",
                "./03_simulation_automation",
                "./04_post_processing",
                "./05_validation_analysis",
                "./06_wind_regime_analysis",
                "./07_visualization",
                "./scripts"
            ],
            "files.autoSave": "afterDelay",
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": "explicit"
            },
            "editor.rulers": [88, 120],
            "[python]": {
                "editor.defaultFormatter": "ms-python.black-formatter",
                "editor.formatOnSave": True
            },
            "terminal.integrated.env.linux": {
                "PYTHONPATH": "${workspaceFolder}:" + str(Path(python_path).parent.parent / "lib/python3.12/site-packages"),
                "PROJECT_ROOT": str(self.project_dir)
            }
        }
        
        with open(vscode_dir / 'settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"✓ Created: .vscode/settings.json")
        
        # launch.json - CORRECTED VERSION
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Current File",
                    "type": "python",
                    "request": "launch",
                    "program": "${file}",
                    "console": "integratedTerminal",
                    "python": python_path,
                    "justMyCode": False,
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}:${env:PYTHONPATH}"
                    }
                },
                {
                    "name": "ERA5 Analysis",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/00_era5_analysis/main.py",
                    "console": "integratedTerminal",
                    "python": python_path,
                    "args": []
                },
                {
                    "name": "Test All",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/scripts/test_all.py",
                    "console": "integratedTerminal",
                    "python": python_path
                }
            ]
        }
        
        with open(vscode_dir / 'launch.json', 'w') as f:
            json.dump(launch_config, f, indent=4)
        print(f"✓ Created: .vscode/launch.json")
        
        # tasks.json
        tasks = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Run ERA5 Analysis",
                    "type": "shell",
                    "command": python_path,
                    "args": ["00_era5_analysis/main.py"],
                    "group": {"kind": "build", "isDefault": True},
                    "presentation": {
                        "reveal": "always",
                        "panel": "dedicated"
                    }
                },
                {
                    "label": "Run Tests",
                    "type": "shell",
                    "command": python_path,
                    "args": ["-m", "pytest", "tests/"],
                    "group": "test"
                },
                {
                    "label": "Format Code",
                    "type": "shell",
                    "command": "black",
                    "args": ["."],
                    "group": {"kind": "build", "isDefault": False}
                }
            ]
        }
        
        with open(vscode_dir / 'tasks.json', 'w') as f:
            json.dump(tasks, f, indent=4)
        print(f"✓ Created: .vscode/tasks.json")
        
        # extensions.json (recommended extensions)
        extensions = {
            "recommendations": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-python.black-formatter",
                "ms-toolsai.jupyter",
                "ms-toolsai.jupyter-keymap",
                "ms-toolsai.jupyter-renderers",
                "ms-toolsai.vscode-jupyter-cell-tags",
                "ms-toolsai.vscode-jupyter-slideshow",
                "KevinRose.vsc-python-indent",
                "njpwerner.autodocstring",
                "Gruntfuggly.todo-tree",
                "eamodio.gitlens"
            ]
        }
        
        with open(vscode_dir / 'extensions.json', 'w') as f:
            json.dump(extensions, f, indent=4)
        print(f"✓ Created: .vscode/extensions.json")
        
        return True

def main():
    """Main function to run setup"""
    
    print("KARIBA WIND ANALYSIS PROJECT SETUP")
    print("=" * 60)
    
    # Create setup instance
    project_dir = '/home/chawas/deployed/charara_01'
    setup = KaribaProjectSetup(project_dir)
    
    # Run setup steps
    try:
        print("\n1. Creating project structure...")
        setup.create_project_structure()
        
        print("\n2. Creating VSCode configuration...")
        setup.create_vscode_config()
        
        print("\n" + "=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
        
        print(f"\nProject created at: {project_dir}")
        print("\nNext steps:")
        print("1. Open VSCode:")
        print(f"   code {project_dir}")
        print("\n2. Select Python interpreter:")
        print("   Press Ctrl+Shift+P → 'Python: Select Interpreter'")
        print(f"   Choose: {setup.config['paths']['python']}")
        print("\n3. Install extensions (recommended):")
        print("   - Python")
        print("   - Pylance")
        print("   - Jupyter")
        print("   - Black Formatter")
        
    except Exception as e:
        print(f"\nError during setup: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())