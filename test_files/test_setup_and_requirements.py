#!/usr/bin/env python3
"""
Test Setup and Requirements for BainGan Application Tests

This file helps set up the testing environment and checks for all required dependencies.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path

# Required packages for testing
REQUIRED_PACKAGES = {
    'unittest': 'Built-in Python testing framework',
    'unittest.mock': 'Built-in mocking library',
    'requests': 'HTTP library for API calls',
    'pandas': 'Data manipulation and analysis',
    'openpyxl': 'Excel file handling',
    'google.generativeai': 'Gemini AI API client',
    'python-dotenv': 'Environment variable management',
    'streamlit': 'Web app framework (mocked in tests)',
    'io': 'Built-in I/O operations',
    'json': 'Built-in JSON handling',
    'datetime': 'Built-in date and time operations',
    'time': 'Built-in time operations',
    'tempfile': 'Built-in temporary file operations'
}

# Test file structure
TEST_FILES = {
    'test_baingan_units.py': 'Unit tests for individual components',
    'test_baingan_integration.py': 'Integration tests for complete workflows',
    'run_all_tests.py': 'Test runner script',
    'advanced_baingan_app.py': 'Main application file (required for imports)'
}

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    if sys.version_info < (3, 7):
        print(f"❌ Python 3.7+ required. Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python {sys.version.split()[0]} (compatible)")
        return True

def check_required_packages():
    """Check if all required packages are available"""
    print("\nChecking required packages...")
    
    missing_packages = []
    available_packages = []
    
    for package, description in REQUIRED_PACKAGES.items():
        try:
            if '.' in package:
                # Handle submodules like unittest.mock
                parent_module = package.split('.')[0]
                importlib.import_module(parent_module)
                submodule = importlib.import_module(package)
            else:
                importlib.import_module(package)
            
            available_packages.append(package)
            print(f"✅ {package} - {description}")
            
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - {description} (MISSING)")
    
    return missing_packages, available_packages

def install_missing_packages(missing_packages):
    """Attempt to install missing packages"""
    if not missing_packages:
        return True
    
    print(f"\nAttempting to install {len(missing_packages)} missing packages...")
    
    # Map package import names to pip install names
    pip_names = {
        'google.generativeai': 'google-generativeai',
        'python-dotenv': 'python-dotenv',
        # Add other mappings as needed
    }
    
    for package in missing_packages:
        pip_name = pip_names.get(package, package)
        
        try:
            print(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
            print(f"✅ Successfully installed {pip_name}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {pip_name}")
            return False
    
    return True

def check_test_files():
    """Check if all required test files exist"""
    print("\nChecking test files...")
    
    missing_files = []
    available_files = []
    
    # Define paths for different file locations
    current_dir = Path('.')
    parent_dir = Path('..')
    
    for filename, description in TEST_FILES.items():
        file_found = False
        
        # Check current directory first
        if (current_dir / filename).exists():
            available_files.append(filename)
            print(f"✅ {filename} - {description}")
            file_found = True
        # For advanced_baingan_app.py, check parent/project_apps directory
        elif filename == 'advanced_baingan_app.py' and (parent_dir / 'project_apps' / filename).exists():
            available_files.append(filename)
            print(f"✅ {filename} - {description} (found in ../project_apps/)")
            file_found = True
        
        if not file_found:
            missing_files.append(filename)
            print(f"❌ {filename} - {description} (MISSING)")
    
    return missing_files, available_files

def create_test_environment_file():
    """Create a sample .env file for testing"""
    env_file = Path('.env.test')
    
    if not env_file.exists():
        print("\nCreating sample test environment file...")
        
        env_content = """# Test Environment Variables for BainGan Application
# Copy this to .env and fill in your actual API keys

# Gemini API Key (required for prompt combination features)
GEMINI_API_KEY=your_gemini_api_key_here

# Test API Configuration (for testing API calls)
TEST_API_URL=https://api.example.com/chat/rag
TEST_API_KEY=your_test_api_key_here

# Test Configuration
TEST_MODE=true
DEBUG_TESTS=false
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Created {env_file}")
        print("   📝 Copy this to .env and update with your actual API keys")
    else:
        print(f"✅ Test environment file already exists: {env_file}")

def run_basic_import_test():
    """Test basic imports to ensure everything works"""
    print("\nRunning basic import test...")
    
    try:
        # Test standard library imports
        import unittest
        import unittest.mock
        import json
        import io
        import datetime
        print("✅ Standard library imports successful")
        
        # Test third-party imports
        import requests
        import pandas as pd
        print("✅ Third-party imports successful")
        
        # Test that we can create basic test structures
        suite = unittest.TestSuite()
        runner = unittest.TextTestRunner()
        mock = unittest.mock.Mock()
        
        print("✅ Test framework components working")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        return False

def print_test_instructions():
    """Print instructions for running tests"""
    print(f"\n{'='*60}")
    print("TEST EXECUTION INSTRUCTIONS")
    print(f"{'='*60}")
    
    print("\n🚀 To run all tests:")
    print("   python run_all_tests.py")
    
    print("\n🔍 To run specific test types:")
    print("   python run_all_tests.py --unit-only")
    print("   python run_all_tests.py --integration-only")
    
    print("\n🎯 To run specific test classes:")
    print("   python run_all_tests.py --class unit:TestAPICall")
    print("   python run_all_tests.py --class integration:TestEndToEndWorkflow")
    
    print("\n📋 To see available test classes:")
    print("   python run_all_tests.py --list-classes")
    
    print("\n📊 To see test coverage information:")
    print("   python run_all_tests.py --coverage")
    
    print("\n⚡ For quick testing (minimal output):")
    print("   python run_all_tests.py --quick")

def main():
    """Main setup function"""
    print(f"{'='*60}")
    print("BAINGAN APPLICATION TEST SETUP")
    print(f"{'='*60}")
    print("Checking test environment and dependencies...\n")
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        sys.exit(1)
    
    # Check required packages
    missing_packages, available_packages = check_required_packages()
    
    # Install missing packages if user agrees
    if missing_packages:
        print(f"\n⚠️  Found {len(missing_packages)} missing packages")
        response = input("Would you like to install them automatically? (y/n): ")
        
        if response.lower().startswith('y'):
            if not install_missing_packages(missing_packages):
                print("\n❌ Setup failed: Could not install required packages")
                print("Please install manually using:")
                for package in missing_packages:
                    pip_name = package.replace('.', '-') if '.' in package else package
                    print(f"   pip install {pip_name}")
                sys.exit(1)
            
            # Re-check packages after installation
            print("\nRe-checking packages after installation...")
            missing_packages, available_packages = check_required_packages()
    
    # Check test files
    missing_files, available_files = check_test_files()
    
    if missing_files:
        print(f"\n⚠️  Missing test files: {', '.join(missing_files)}")
        print("Please ensure all test files are in the same directory")
    
    # Create test environment file
    create_test_environment_file()
    
    # Run basic import test
    if not run_basic_import_test():
        print("\n❌ Setup failed: Basic import test failed")
        sys.exit(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SETUP SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Python version: Compatible")
    print(f"✅ Available packages: {len(available_packages)}")
    print(f"✅ Available test files: {len(available_files)}")
    print(f"✅ Basic imports: Working")
    
    if missing_packages:
        print(f"⚠️  Missing packages: {len(missing_packages)}")
    if missing_files:
        print(f"⚠️  Missing files: {len(missing_files)}")
    
    if not missing_packages and not missing_files:
        print("\n🎉 Setup complete! Ready to run tests.")
        print_test_instructions()
    else:
        print("\n⚠️  Setup completed with warnings.")
        print("Some components may not work properly.")
        if missing_files:
            print("Missing test files should be created before running tests.")

if __name__ == '__main__':
    main()