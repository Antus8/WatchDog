## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
# Import the setup function from distutils to configure package installation
from distutils.core import setup
# Import generate_distutils_setup from catkin_pkg to generate ROS-compatible setup arguments
from catkin_pkg.python_setup import generate_distutils_setup

# Generate distutils setup arguments by parsing the package.xml file
# This automatically extracts metadata like name, version, dependencies, etc.
setup_args = generate_distutils_setup(
    packages=['package'],  # List of Python packages to include in the distribution
    package_dir={'': 'src'},  # Map package root directory to 'src' folder
)
# Execute the setup with the generated arguments to configure the package
setup(**setup_args)