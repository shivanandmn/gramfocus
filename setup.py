from setuptools import setup, find_packages

setup(
    name="gramfocus",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # API and Web Server
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        
        # LLM Services
        "openai>=1.3.5",
        "google-generativeai>=0.3.1",
        
        # Speech-to-Text
        "google-cloud-speech>=2.21.0",
        
        # File Handling
        "aiofiles>=23.2.1",
        
        # Configuration and Validation
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.2",
        "pydantic-settings>=2.1.0",
        
        # JSON Processing
        "json-repair>=0.30.2",
    ]
)
