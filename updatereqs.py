import re

def parse_requirements(file_path):
    """Parse the requirements file and return a dictionary with package names as keys and versions as values."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    requirements = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            match = re.match(r'([a-zA-Z0-9\-_.]+)([=><!~]*)(.+)?', line)
            if match:
                package, operator, version = match.groups()
                requirements[package.lower()] = operator + version if version else ""
    return requirements

def update_requirements(reqs_path, requirements_path, output_path):
    reqs = parse_requirements(reqs_path)
    requirements = parse_requirements(requirements_path)
    
    updated_requirements = []
    for package, version in requirements.items():
        if package in reqs:
            updated_requirements.append(f"{package}{reqs[package]}")
        else:
            updated_requirements.append(f"{package}{version}")
    
    with open(output_path, 'w') as file:
        for requirement in updated_requirements:
            file.write(requirement + '\n')

# Define the paths to your files
reqs_path = 'reqs.txt'
requirements_path = 'requirements.txt'
output_path = 'requirements.txt'

# Update the requirements
update_requirements(reqs_path, requirements_path, output_path)