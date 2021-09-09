# How to contribute

All contributions, ideas and bug reports are more than welcome! 
We encourage you to open an [issue](https://github.com/artefactory-global/streamlit_prophet/issues) for any change you would like to make on this project.

## Workflow

If you wish to contribute, please follow this process:

* Fork the main branch of the repository.
* Clone your fork locally.
* Set up your environment (see `Dependencies` section below).
* Commit your work after ensuring it meets the guidelines described below (code style, checks).
* Push to your fork.
* Open a pull request from your fork back to the original main branch.

## Dependencies

We use `poetry` to manage the [dependencies](https://github.com/python-poetry/poetry).
If you dont have `poetry` installed, you should run the command below.

```bash
make download-poetry; export PATH="$HOME/.poetry/bin:$PATH"
```

To install dependencies and prepare [`pre-commit`](https://pre-commit.com/) hooks you would need to run `install` command:

```bash
make install
```

To activate your `virtualenv` run `poetry shell`.

## Code style

After you run `make install` you can execute the automatic code formatting.

```bash
make format-code
```

### Checks

Many checks are configured for this project. Command `make check-style` will run black diffs, darglint docstring style and mypy.
The `make check-safety` command will look at the security of your code.

You can also use `STRICT=1` flag to make the check be strict.

### Before submitting

Before submitting your code please do the following steps:

1. Add any changes you want
1. Add tests for the new changes
1. Edit documentation if you have changed something significant
1. Run `make format-code` to format your changes.
1. Run `STRICT=1 make check-style` to ensure that types and docs are correct
1. Run `STRICT=1 make check-safety` to ensure that security of your code is correct

## Other help

You can also contribute by spreading a word about this library.
We would be very interested to hear how you are using the app and what are your best practices.
