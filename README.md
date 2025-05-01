# STT811 Alzheimer's Dataset Exploration Disease Prediction

### Files:

📁 `cache`
- Folder holding cached modeling results and model importances. Due to streamlit data limitations, caches are used for saving time and resources.

📁 `data`
- Folder holding our dataset(s).

📁 `old_notebooks`
- Folder older notebooks from our individual explorations later combined into 'master_notebook.ipynb'.

📁 `pages`
- Folder holding our streamlit individual page source code files.

📄 `app.py`
- Head source file for our streamlit application.

📄 `preprocessing.py`
- Source code for our preprocessing step for the alzheimers data.

📄 `util.py`
- Global helper functions. May be refactored later to be more specified.

📓 `master_notebook.ipynb`
- Conglomerate master notebook for exploration of the data in an interactive manner.

📄 `requirements.txt`
- Lightweight environment dependencies file.


### Local Development:

To develop locally, clone the repository, install dependencies from requirements.txt into a python environment of your choice, and run command:

`streamlit run app.py`

or the long form:

`python -m streamlit run app.py`
