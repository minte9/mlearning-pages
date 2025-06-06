from setuptools import setup, find_packages

setup(
    name="deploy-ai-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.10.0",
        "python-dotenv>=1.0.0"
    ],
    entry_points={
        "console_scripts": [
            "deployai=deploy_ai_agent:main"
        ]
    },
    author="Catalin Prescure",
    description="AI-powered CLI agent to deploy Git and FTP changes from natural language commands.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_date=True,
    python_requires=">=3.7",
)