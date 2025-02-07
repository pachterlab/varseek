import toml
from setuptools import find_packages, setup

# from varseek.__init__ import __version__, __author__, __email__


def read(path):
    with open(path, "r") as f:
        return f.read()
    

def get_metadata_from_pyproject():
    with open("pyproject.toml", "r") as f:
        pyproject_data = toml.load(f)
        project_data = pyproject_data["project"]
        
        # Extract version, author, and other metadata
        version = project_data["version"]
        description = project_data.get("description", "")
        author = project_data["authors"][0]["name"]
        author_email = project_data["authors"][0]["email"]
        maintainer = project_data["maintainers"][0]["name"] if "maintainers" in project_data else author
        maintainer_email = project_data["maintainers"][0]["email"] if "maintainers" in project_data else author_email
        
        python_version = project_data.get("requires-python", "")

        install_requires = [
            f"{package}{version}"
            for package, version in pyproject_data.get("project", {}).get("dependencies", {}).items()
        ]

        extras_require = pyproject_data.get("project", {}).get("optional-dependencies", {})
        
        return version, description, author, author_email, maintainer, maintainer_email, python_version, install_requires, extras_require


long_description = read("README.md")
version, description, author, author_email, maintainer, maintainer_email, python_version, install_requires, extras_require = get_metadata_from_pyproject()

setup(
    name="varseek",
    version=version,
    license="BSD-2",
    author=author,
    author_email=author_email,
    maintainer=maintainer,
    maintainer_email=maintainer_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    packages=find_packages(include=["varseek", "varseek.*"]),
    include_package_data=True,
    python_requires=python_version,
    install_requires=install_requires,
    setup_requires=install_requires,
    extras_require=extras_require,
    url="https://github.com/pachterlab/varseek",
    keywords="varseek",
    entry_points={
        "console_scripts": ["vk=varseek.main:main"],
    },
    classifiers=[
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Utilities",
    ]
)
