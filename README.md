<div align="center">

![](streamlit_prophet/references/logo.png)

[![CI status](https://github.com/artefactory-global/streamlit_prophet/actions/workflows/ci.yml/badge.svg?branch%3Amain&event%3Apush)](https://github.com/artefactory-global/streamlit_prophet/actions/workflows/ci.yml?query=branch%3Amain)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)](#supported-python-versions)
[![Dependencies Status](https://img.shields.io/badge/dependabots-active-informational.svg)](https://github.com/artefactory-global/streamlit_prophet/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-informational.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory-global/streamlit_prophet/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%F0%9F%9A%80-semantic%20versions-informational.svg)](https://github.com/artefactory-global/streamlit_prophet/releases)
[![License](https://img.shields.io/badge/License-MIT-informational.svg)](https://github.com/artefactory-global/streamlit_prophet/blob/main/LICENSE)

Deploy a [Streamlit](https://streamlit.io/) app to visually train, evaluate and optimize a [Prophet](https://facebook.github.io/prophet/) forecasting model

</div>


## 💻 Supported Python Versions

<p>Main version supported : <strong>3.7</strong> <br>
Other supported versions : <strong>3.8</strong> & <strong>3.9</strong> </p>

Please make sure you have one of these versions installed to be able to run the app locally.


## ⚙️ Installation


### Create a virtual environment (optional)
If you want to avoid any dependencies issues, first create and activate a new virtual environment.

For example with conda:
```bash
pip install conda; conda create -n streamlit_prophet python=3.7; conda activate streamlit_prophet
```

Or with virtualenv:
```bash
pip install virtualenv; python3.7 -m virtualenv streamlit_prophet --python=python3.7; source streamlit_prophet/bin/activate
```


### Install package
Install the package from the main branch of this repository:

```bash
pip install git+ssh://git@github.com/artefactory-global/streamlit_prophet.git@main
```


## 📈 Usage

Once installed, run the following command from CLI to access the app:

```bash
streamlit_prophet deploy dashboard
```

Now you can train, evaluate and optimize a forecasting model in just a few clicks.
All you have to do is to upload a time series dataset, that contains a date column, a target column and optionally some features, like on the example below:

![](streamlit_prophet/references/input_format.png)

Then, follow the guidelines in the sidebar to:

* <strong>Prepare data</strong>: Filter, aggregate, resample and/or clean your dataset.
* <strong>Choose model parameters</strong>: Default parameters are available but you can tune them.
Look at the tooltips to understand how each parameter is impacting forecasts.
* <strong>Select evaluation method</strong>: Define the evaluation process, the metrics and the granularity to
assess your model performance.
* <strong>Make a forecast</strong>: Make a forecast on future dates that are not included in your dataset,
with the model previously trained.

Once you are satisfied, click on "save experiment" to download all plots and data locally.

https://user-images.githubusercontent.com/56996548/126762714-f2d3f3a1-7098-4a86-8c60-0a69d0f913a7.mp4


## 🛠️ How to contribute ?

If you want to contribute to the development of this package:

1. Clone the repository:

```bash
git clone git@github.com:artefactory-global/streamlit_prophet.git
```

2. If you don't have `Poetry` installed, run:

```bash
make download-poetry; export PATH="$HOME/.poetry/bin:$PATH"
```

3. Initialize poetry and install `pre-commit` hooks:

```bash
make install
```

And you are ready to develop ! For more information, see [`CONTRIBUTING`](https://github.com/artefactory-global/streamlit_prophet/blob/main/CONTRIBUTING.md) instructions.
If you wish to containerize the app, see [`DOCKER`](https://github.com/artefactory-global/streamlit_prophet/blob/main/DOCKER.md) instructions.

