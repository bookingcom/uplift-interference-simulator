from setuptools import setup, find_namespace_packages

# Read requirements.txt and load all requirements into a list
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="infsim",
    version="0.1",
    description="A toolset to simulate in-search interference for incentive and discount treatments",
    author="Bram van den Akker",
    author_email="bram.vandenakker@booking.com",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    # install_requires=requirements,  # Use the list of requirements as the install requires
)
