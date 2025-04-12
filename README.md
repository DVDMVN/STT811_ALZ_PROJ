# STT811 Alzheimer's Dataset Exploration Disease Prediction

### Files:

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

📔 `download_dataset.ipynb`
- Some code to download the dataset using the python kaggle API. Since the data is small enough, the data can be uploaded to GitHub without replication issue, so this notebook may not be used.

📓 `master_notebook.ipynb`
- Conglomerate master notebook for exploration of the data in an interactive manner.

📄 `requirements.txt`
- Lightweight environment dependencies file.


### Local Development:

To develop locally, clone the repository, install dependencies from requirements.txt into a python environment of your choice, and run command:

`streamlit run app.py`

or the long form:

`python -m streamlit run app.py`
