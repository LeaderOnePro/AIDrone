from setuptools import setup, find_packages

setup(
    name="aidrone",
    version="0.1.0",
    description="AIDrone - AI-powered drone control and mission planning",
    author="AIDrone Team",
    packages=find_packages(),
    install_requires=[
        "dronekit",
        "smolagents",
        "streamlit",
        "huggingface_hub",
        "python-dotenv",
    ],
) 