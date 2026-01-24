Create and activate a venv.

    cd .\developments\python\mlearning\
    python -m venv .\venv
    .\venv\Scripts\activate
    (venv) PS

Requirements file for packages.

    numpy>=1.24,<3.0
    pandas>=2.0,<3.0
    matplotlib>=3.7,<4.0
    scikit-learn>=1.3,<2.0
    icecream>=2.1,<3.0

Upgrade pip/setuptools/wheel (helps with binary wheels).

    python -m pip install --upgrade pip setuptools wheel

Install your requirements.

    pip install -r requirements.txt
