# Repository Guidelines

## Project Structure & Module Organization
This repository is organized around machine learning study materials rather than a single application. Use `notebook/` for exploratory Jupyter notebooks, `sample/` for exercise or answer notebooks, `talk/` for presentation-specific notebooks, and `dataset/` for local data files such as `dataset/breast_cancer_data.csv`. Written explanations live in `markdown/` and `mindmap/`, while generated exports such as PDFs belong in `pdf/`. The `materials/` directory is a separate Slidev deck with Vue components in `materials/components/` and slides in `materials/slides.md`.

## Build, Test, and Development Commands
Set up Python dependencies with `uv sync` from the repository root. Run the small entry script with `uv run python main.py`. For notebook work, launch Jupyter with `uv run jupyter notebook`. In `materials/`, install frontend dependencies with `npm install`, start the deck locally with `npm run dev`, build static slides with `npm run build`, and export presentation assets with `npm run export`.

## Coding Style & Naming Conventions
Use 4-space indentation in Python and follow PEP 8 naming: `snake_case` for functions, variables, and notebook filenames where possible. Keep notebook names topic-based, for example `logistic_regression.ipynb` or `ridge_regression.ipynb`. In Vue and Slidev files, follow the existing component naming pattern such as `Counter.vue`.

## Testing Guidelines
There is no dedicated automated test suite yet. Validate Python changes by re-running the affected notebook cells end to end and checking generated plots, metrics, and exports. For `materials/`, use `npm run build` as the minimum verification step before submitting changes. If you add reusable Python code, include `pytest` tests under a new `tests/` directory and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Recent commits use short, descriptive messages in Japanese that explain the intent of the change. Follow that style, for example: `ロジスティック回帰の説明を更新` or `決定境界のノートブックを追加`. Keep commits focused on one topic. Pull requests should include a brief summary, affected paths, any dataset or notebook outputs that changed, and screenshots when slide visuals in `materials/` are updated.
