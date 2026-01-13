# Quantum Market Forecasting Pipeline

## Project Overview

This project builds an end-to-end, **open-source AI pipeline** for generating and explaining **daily stock price movement predictions** for companies in the **Quantum Computing industry**.

At a high level, the system:

1. Ingests **daily market data** (prices, volume, etc.)
2. Ingests **news and social media data**
3. Applies **financial sentiment analysis**
4. Combines sentiment signals with **traditional forecasting models**
5. Generates **price movement predictions**
6. Stores predictions and supporting evidence in a database
7. Exposes results via a **dashboard**, including:
   - The prediction
   - Confidence / uncertainty (where applicable)
   - A plain-language explanation of *why* the system made that prediction

This repository is structured to support **collaborative development**, **reproducible research**, and eventual deployment as a working product.

---

## Repository Structure

Below is a brief guide to what each top-level directory is for.

### `docs/`
Project documentation.
- `architecture/`: system design, data flow, agent interactions
- `decisions/`: architecture decision records (why we chose certain tools)
- `runbooks/`: how to run, debug, or deploy the system

If you’re new to the project, start here.

---

### `configs/`
Configuration files (YAML).
- Environment settings (dev vs prod)
- Model and pipeline parameters
- Prompt templates for LLM-based agents

Avoid hardcoding parameters elsewhere — configs should live here.

---

### `data/`
Data directory **(mostly gitignored)**.
- `raw/`: unprocessed data pulled from external sources
- `external/`: third-party datasets
- `interim/`: partially processed data
- `processed/`: clean datasets ready for modeling
- `features/`: model-ready feature tables
- `labels/`: targets / outcomes used for training

See `data/README.md` for data handling rules.

---

### `notebooks/`
Exploratory and research notebooks.
- Used for prototyping, analysis, and validation
- Not used directly in production pipelines

Folder numbering reflects the typical workflow order.

---

### `src/qsf/`
The **core Python package** (all production code lives here).

Key submodules:

- `common/`  
  Shared utilities, constants, schemas, logging, and helper functions

- `ingestion/`  
  Code for pulling in:
  - Market data
  - News articles
  - Social media content

- `nlp/`  
  Text cleaning, embeddings, sentiment models, and aggregation logic

- `features/`  
  Feature engineering that combines market data + sentiment signals

- `forecasting/`  
  Time-series and ML models for predicting price movements

- `backtesting/`  
  Walk-forward evaluation, leakage checks, and performance metrics

- `agents/`  
  Multi-agent orchestration layer used to:
  - Coordinate pipeline steps
  - Generate human-readable explanations
  - Retrieve supporting evidence

- `pipelines/`  
  End-to-end runnable workflows (ETL, training, inference, backtesting)

- `api/`  
  FastAPI application exposing predictions and explanations to the dashboard

---

### `tests/`
Automated tests.
- `unit/`: individual functions and components
- `integration/`: multi-step pipeline tests
- `fixtures/`: small test datasets

All new production code should include tests where feasible.

---

### `scripts/`
Helper scripts for:
- One-off data pulls
- Local pipeline runs
- Developer utilities

---

### `infrastructure/`
Deployment and infrastructure code.
- `docker/`: Dockerfiles and docker-compose
- `terraform/`: (optional) cloud infrastructure
- `github/workflows/`: CI/CD pipelines

---

### `reports/`
Generated outputs and artifacts.
- Backtest results
- Weekly summaries
- Figures and plots used in presentations

---

## How to Contribute (High-Level)

- Use **branches + pull requests**
- Keep experimental work in **notebooks/**
- Put reusable logic in **src/qsf/**
- Document assumptions and decisions
- When in doubt, ask before refactoring shared code

---

## Project Status

This is an **active, collaborative research and engineering project**.  
Expect some interfaces to evolve as we learn what works.

