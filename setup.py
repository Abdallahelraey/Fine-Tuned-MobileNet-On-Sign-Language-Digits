from setuptools import setup, find_packages

setup(
    name="Fine-Tuned-MobileNet-On-Sign-Language-Digits",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.16.1",
        "numpy==1.23.5",
        "matplotlib",
        "scikit-learn",
        "pytest",
        "pyyaml",
        "python-dotenv",
        "pydantic",
        "pydantic-settings",
        "python-dotenv==1.0.1",
        "mkdocs==1.6.0",
        "mkdocs-material",
        "mkdocstrings ==0.20.0",
        "mkdocstrings-python == 0.8.3",
        # "mesop",
        # "fastapi==0.110.2",
        # "uvicorn[standard]==0.29.0"
        
    ]
)

