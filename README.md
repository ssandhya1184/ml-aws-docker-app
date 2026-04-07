# ml-aws-docker-app
Deploy a simple ML model in aws using docker

# ML AWS Docker App

## Features
- ML model training
- FastAPI inference
- Dockerized deployment
- AWS EC2 ready

## Setup

uv venv
uv sync

## Run

python src/pipeline/train_pipeline.py
uvicorn src.api.main:app --reload


1. Create a new repo. CLone in Local
2. Add required folders
3. Update gitignore
4. uv init
   uv venv
   uv add - required libraries

   ml-aws-docker-app/
│
├── .venv/                 # virtual env (ignored in git)
├── src/                   # main source code
│   ├── __init__.py
│   ├── components/        # reusable modules
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/          # training/inference pipelines
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── api/               # FastAPI app
│   │   └── main.py
│   │
│   ├── utils/             # helper functions
│   │   └── common.py
│   │
│   └── logger.py
│
├── notebooks/             # Jupyter notebooks (EDA)
│   └── experiments.ipynb
│
├── artifacts/             # trained models, outputs
│   └── model.pkl
│
├── tests/                 # unit tests
│   └── test_pipeline.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock                # auto-generated lock file
