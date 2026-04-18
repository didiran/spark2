from setuptools import setup, find_packages

setup(
    name="fraud-detection-api",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "psycopg2-binary",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "python-multipart",
        "kafka-python",
        "pydantic",
        "email-validator",
        "jinja2",
        "bcrypt"
    ],
    python_requires=">=3.8",
)
