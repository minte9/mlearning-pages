from setuptools import setup
from pathlib import Path

setup(
    name="deploy-ai-agent",
    version="0.1.0",
    py_modules=["deploy_ai_agent"],
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
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    include_package_data=True,
    python_requires=">=3.7",
)