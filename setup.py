from setuptools import setup, find_packages

setup(
    name="deepdrone-old",
    version="1.0.0",
    description="deepdrone-old - AI-powered drone control and mission planning",
    author="deepdrone-old Team",
    packages=find_packages(),
    install_requires=[
        "dronekit",
        "smolagents",
        "streamlit",
        "huggingface_hub",
        "python-dotenv",
    ],
) 