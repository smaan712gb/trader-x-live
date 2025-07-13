#!/usr/bin/env python3
"""
Environment setup script for Trader-X
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_conda():
    """Check if conda is available"""
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def setup_environment():
    """Set up the Trader-X environment"""
    print("=" * 60)
    print("TRADER-X ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Check if conda is available
    if not check_conda():
        print("✗ Conda is not available. Please install Miniconda or Anaconda first.")
        return False
    
    print("✓ Conda is available")
    
    # Check if environment already exists
    env_name = "Trader-X"
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    
    if env_name in result.stdout:
        print(f"✓ Environment '{env_name}' already exists")
        
        # Ask if user wants to recreate
        response = input(f"Do you want to recreate the environment? (y/N): ").strip().lower()
        if response == 'y':
            if not run_command(f"conda env remove -n {env_name} -y", f"Removing existing environment '{env_name}'"):
                return False
        else:
            print("Using existing environment")
            return True
    
    # Create conda environment
    if not run_command(f"conda create -n {env_name} python=3.11 -y", f"Creating conda environment '{env_name}'"):
        return False
    
    # Activate environment and install packages
    activate_cmd = f"conda activate {env_name}"
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("✗ requirements.txt not found")
        return False
    
    # Install packages
    install_cmd = f"{activate_cmd} && pip install -r requirements.txt"
    if not run_command(install_cmd, "Installing Python packages"):
        return False
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"To activate the environment, run: conda activate {env_name}")
    print("To test the system, run: python test_system.py")
    print("=" * 60)
    
    return True

def main():
    """Main function"""
    if not setup_environment():
        print("\n✗ Setup failed!")
        sys.exit(1)
    
    print("\n✓ Setup completed successfully!")

if __name__ == "__main__":
    main()
