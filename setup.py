from setuptools import setup, find_packages
import os

# Function to list all MATLAB files in a directory
def list_matlab_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.m'):
                # Here, we strip the 'src/' part, as the paths in package_data should be relative to the package
                yield os.path.relpath(os.path.join(root, file), 'src')

# Specify the directory containing MATLAB files
matlab_files_directory = 'src/ebFRET'

# List all MATLAB files
matlab_files = list(list_matlab_files(matlab_files_directory))

setup(
    name='DeepGapSeq',
    version='0.1',
    packages=find_packages(),
    package_data={
        # Include MATLAB files in the package
        'DeepGapSeq': matlab_files,
    },
)
