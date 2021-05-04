# streamlit_prophet

<div align="center">

[![CI status](https://github.com/artefactory/streamlit_prophet/actions/workflows/ci.yml/badge.svg?branch%3Amain&event%3Apush)](https://github.com/artefactory/streamlit_prophet/actions/workflows/ci.yml?query=branch%3Amain)
[![CD status](https://github.com/artefactory/streamlit_prophet/actions/workflows/cd.yml/badge.svg?event%3Arelease)](https://github.com/artefactory/streamlit_prophet/actions/workflows/cd.yml?query=event%3Arelease)
[![Python Version](https://img.shields.io/badge/Python-3.7-informational.svg)](#supported-python-versions)
[![Dependencies Status](https://img.shields.io/badge/dependabots-active-informational.svg)](https://github.com/artefactory/streamlit_prophet}/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-informational.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/streamlit_prophet}/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%F0%9F%9A%80-semantic%20versions-informational.svg)](https://github.com/artefactory/streamlit_prophet/releases)
[![Documentation](https://img.shields.io/badge/doc-sphinx-informational.svg)](https://github.com/artefactory/streamlit_prophet}/tree/main/docs)
[![License](https://img.shields.io/badge/License-Private%20Use-informational.svg)](https://github.com/artefactory/streamlit_prophet}/blob/main/LICENSE)

`streamlit_prophet` is a Python cli/package

</div>

## TL;DR

```bash
pip install git+ssh://git@github.com/artefactory/streamlit_prophet.git@main
```

## Supported Python Versions

Main version supported : `3.7`

Other supported versions :


## Installation

### Public Package

Install the package from PyPi if it is publicly available :

```bash
pip install -U streamlit_prophet
```

or with `Poetry`

```bash
poetry add streamlit_prophet
```

### Private Package

Install the package from a private Github repository from the main branch:

```bash
pip install git+ssh://git@github.com/artefactory/streamlit_prophet.git@main
```

Or from a known tag:

```bash
pip install git+ssh://git@github.com/artefactory/streamlit_prophet.git@v0.1.0
```

Or from the latest Github Release using [personal access tokens](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token)

```bash
# Set the GITHUB_TOKEN environment variable to your personal access token
latest_release_tag=$(curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/repos/artefactory/streamlit_prophet/releases/latest | jq -r ".tag_name")
pip install git+ssh://git@github.com/artefactory/streamlit_prophet.git@${latest_release_tag}
```

## Usage

Once installed, you can run commands like :

```bash
streamlit_prophet --help
```

```bash
streamlit_prophet dialogs hello --name Roman
```

```bash
streamlit_prophet dialogs clock --color blue
```

or if installed with `Poetry`:

```bash
poetry run streamlit_prophet --help
```

```bash
poetry run streamlit_prophet dialogs hello --name Roman
```

## Local setup

If you want to contribute to the development of this package, 

1. Clone the repository :

```bash
git clone git@github.com:artefactory/streamlit_prophet.git
```

2. If you don't have `Poetry` installed, run:

```bash
make download-poetry; source "$HOME/.poetry/env"
```

3. Initialize poetry and install `pre-commit` hooks:

```bash
make install
```

And you are ready to develop !

### Initial Github setting up

Can be automatically done directly after using the [`cookiecutter` template](https://github.com/artefactory/ppt) :

```bash
make github-install
```

Otherwise, see :
- [Stale bot](https://github.com/apps/stale) for automatic issue closing.
- [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) to deploy your documentation from the `gh-pages` branch.
- [GitHub Branch Protection](https://docs.github.com/en/github/administering-a-repository/about-protected-branches) to secure your `main` branch.

### Poetry

All manipulations with dependencies are executed through Poetry. If you're new to it, look through [the documentation](https://python-poetry.org/docs/).

<details>
<summary>Notes about Poetry commands</summary>
<p>

Here are some examples of Poetry [commands](https://python-poetry.org/docs/cli/#commands) :

- `poetry add numpy`
- `poetry run pytest`
- `poetry build`
- etc

</p>
</details>

### Building your package

Building a new version of the application contains steps:

- Bump the version of your package `poetry version <version>`. You can pass the new version explicitly, or a rule such as `major`, `minor`, or `patch`. For more details, refer to the [Semantic Versions](https://semver.org/) standard.
- Make a commit to `GitHub`.
- Create a `GitHub release`.
- And... publish  only if you want to make your package publicly available on PyPi 🙂 `poetry publish --build`

Packages:

- [`Typer`](https://github.com/tiangolo/typer) is great for creating CLI applications.
- [`Rich`](https://github.com/willmcgugan/rich) makes it easy to add beautiful formatting in the terminal.

## 🚀 Features

- Support for `Python 3.7`
- [`Poetry`](https://python-poetry.org/) as the dependencies manager. See configuration in [`pyproject.toml`](https://github.com/artefactory/streamlit_prophet}/blob/main/pyproject.toml) and [`setup.cfg`](https://github.com/artefactory/streamlit_prophet}/blob/main/setup.cfg).
- Power of [`black`](https://github.com/psf/black), [`isort`](https://github.com/timothycrosley/isort) and [`pyupgrade`](https://github.com/asottile/pyupgrade) formatters.
- Ready-to-use [`pre-commit`](https://pre-commit.com/) hooks with formatters above.
- Type checks with [`mypy`](https://mypy.readthedocs.io).
- Testing with [`pytest`](https://docs.pytest.org/en/latest/).
- Docstring checks with [`darglint`](https://github.com/terrencepreilly/darglint).
- Security checks with [`safety`](https://github.com/pyupio/safety), [`bandit`](https://github.com/PyCQA/bandit) and [`anchore`](https://github.com/anchore/scan-action).
- Secrets scanning with [`gitleaks`](https://github.com/zricethezav/gitleaks)
- Well-made [`.editorconfig`](https://github.com/artefactory/streamlit_prophet}/blob/main/.editorconfig), [`.dockerignore`](https://github.com/artefactory/streamlit_prophet}/blob/main/.dockerignore), and [`.gitignore`](https://github.com/artefactory/streamlit_prophet}/blob/main/.gitignore). You don't have to worry about those things.

For building and deployment:

- `GitHub` integration.
- [`Makefile`](https://github.com/artefactory/streamlit_prophet}/blob/main/Makefile#L89) for building routines. Everything is already set up for security checks, codestyle checks, code formatting, testing, linting, docker builds, etc. More details at [Makefile summary](#makefile-usage)).
- [Dockerfile](https://github.com/artefactory/streamlit_prophet}/blob/main/docker/Dockerfile) for your package.
- `Github Actions` with predefined [build workflow](https://github.com/artefactory/streamlit_prophet}/blob/main/.github/workflows/build.yml) as the default CI/CD.- `Github Pages` documentation built with Sphinx updated automatically when [releasing the package](https://github.com/artefactory/streamlit_prophet}/blob/main/.github/workflows/cd.yml).
- `Github Packages` with Github Container Registry updated automatically when [releasing the package](https://github.com/artefactory/ppt/blob/main/%7B%7B%20cookiecutter.project_name.lower().replace('%20'%2C%20'_')%20%7D%7D/.github/workflows/cd.yml).
- Automatic drafts of new releases with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). It creates a list of changes based on labels in merged `Pull Requests`. You can see labels (aka `categories`) in [`release-drafter.yml`](https://github.com/artefactory/streamlit_prophet}/blob/main/.github/release-drafter.yml). Works perfectly with [Semantic Versions](https://semver.org/) specification.

For creating your open source community:

- Ready-to-use [Pull Requests templates](https://github.com/artefactory/streamlit_prophet}/blob/main/.github/PULL_REQUEST_TEMPLATE.md) and several [Issue templates](https://github.com/artefactory/streamlit_prophet}/tree/main/.github/ISSUE_TEMPLATE).
- Files such as: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` are generated automatically.
- [`Stale bot`](https://github.com/apps/stale) that closes abandoned issues after a period of inactivity. (You will only [need to setup free plan](https://github.com/marketplace/stale)). Configuration is [here](https://github.com/artefactory/streamlit_prophet}/blob/main/.github/.stale.yml).
- [Semantic Versions](https://semver.org/) specification with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter).

### Makefile usage

[`Makefile`](https://github.com/artefactory/streamlit_prophet}/blob/main/Makefile) contains many functions for fast assembling and convenient work.

<details>
<summary>1. Download Poetry</summary>
<p>

```bash
make download-poetry; source "$HOME/.poetry/env"
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

```bash
make install
```

If you do not want to install pre-commit hooks, run the command with the NO_PRE_COMMIT flag:

```bash
make install NO_PRE_COMMIT=1
```

</p>
</details>

<details>
<summary>3. Check the security of your code</summary>
<p>

```bash
make check-safety
```

This command launches a `Poetry` and `Pip` integrity check as well as identifies security issues with `Safety` and `Bandit`. By default, the build will not crash if any of the items fail. But you can set `STRICT=1` for the entire build, or you can configure strictness for each item separately.

```bash
make check-safety STRICT=1
```

or only for `safety`:

```bash
make check-safety SAFETY_STRICT=1
```

multiple

```bash
make check-safety PIP_STRICT=1 SAFETY_STRICT=1
```

> List of flags for `check-safety` (can be set to `1` or `0`): `STRICT`, `POETRY_STRICT`, `PIP_STRICT`, `SAFETY_STRICT`, `BANDIT_STRICT`.

</p>
</details>

<details>
<summary>4. Check the codestyle</summary>
<p>

The command is similar to `check-safety` but to check the code style, obviously. It uses `Black`, `Darglint`, `Isort`, and `Mypy` inside.

```bash
make check-style
```

It may also contain the `STRICT` flag.

```bash
make check-style STRICT=1
```

> List of flags for `check-style` (can be set to `1` or `0`): `STRICT`, `BLACK_STRICT`, `DARGLINT_STRICT`, `ISORT_STRICT`, `MYPY_STRICT`.

</p>
</details>

<details>
<summary>5. Run all the codestyle formatters</summary>
<p>

Codestyle uses `pre-commit` hooks, so ensure you've run `make install` before.

```bash
make format-code
```

</p>
</details>

<details>
<summary>6. Run tests</summary>
<p>

```bash
make test
```

</p>
</details>

<details>
<summary>7. Run all the linters</summary>
<p>

```bash
make lint
```

the same as:

```bash
make test && make check-safety && make check-style
```

> List of flags for `lint` (can be set to `1` or `0`): `STRICT`, `POETRY_STRICT`, `PIP_STRICT`, `SAFETY_STRICT`, `BANDIT_STRICT`, `BLACK_STRICT`, `DARGLINT_STRICT`, `PYUPGRADE_STRICT`, `ISORT_STRICT`, `MYPY_STRICT`.

</p>
</details>

<details>
<summary>8. Build docker</summary>
<p>

```bash
make docker
```

which is equivalent to:

```bash
make docker VERSION=latest
```

More information [here](https://github.com/artefactory/streamlit_prophet}/tree/main/docker).

</p>
</details>

<details>
<summary>9. Cleanup docker</summary>
<p>

```bash
make clean_docker
```

or to remove all build

```bash
make clean
```

More information [here](https://github.com/artefactory/streamlit_prophet}/tree/main/docker).

</p>
</details>


<details>
<summary>10. Activate virtualenv</summary>
<p>

```bash
poetry shell
```

To deactivate the virtual environment :

```bash
deactivate
```

</p>
</details>

## 📈 Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/artefactory/streamlit_prophet}/releases) page.

We follow [Semantic Versions](https://semver.org/) specification.

We use [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). As pull requests are merged, a draft release is kept up-to-date listing the changes, ready to publish when you’re ready. With the categories option, you can categorize pull requests in release notes using labels.

For Pull Requests, these labels are configured, by default:

|               **Label**               |  **Title in Releases**  |
| :-----------------------------------: | :---------------------: |
|       `enhancement`, `feature`        |       🚀 Features       |
| `bug`, `refactoring`, `bugfix`, `fix` | 🔧 Fixes & Refactoring  |
|       `build`, `ci`, `testing`        | 📦 Build System & CI/CD |
|              `breaking`               |   💥 Breaking Changes   |
|            `documentation`            |    📝 Documentation     |
|            `dependencies`             | ⬆️ Dependencies updates |


GitHub creates the `bug`, `enhancement`, and `documentation` labels automatically. Dependabot creates the `dependencies` label. Create the remaining labels on the Issues tab of the GitHub repository, when needed.## 📃 Citation

```
@misc{streamlit_prophet,
  author = {artefactory},
  title = {`streamlit_prophet` is a Python cli/package},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/artefactory/streamlit_prophet}}}
}
```

## Credits

This project was generated with [`ppt`](https://github.com/artefactory/ppt).
