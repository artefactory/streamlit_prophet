# Artefact's Boilerplate for  DataScience Python projects 

This repository is a boilerplate repository designed to be used when starting a new project to help kickstart things easily.

To create a new repository based on this one please use the "create from a template" feature (see [Github's documentation](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template)).

There will be more than you need for your project so feel free to drop what you don't need after you initialized your repo with this one.

## Organisation of the repo

### Folders

    The folder name should be self sufficient, however some extra details:
    - bin: This is the folder where you store your executable, that could be main python scripts, or bash ones.
    - lib: this is where you store the main libraries used within your project. 
    - data: Separated in 3 folder to start: Raw data, intermediate, and processed.
    - doc: Sphinx template to generate code documentation
    - references: all the written documentation (functional, features) that is not sphinx generated 

### Requirements

You should always have requirements to your project to ensure reproductibility.

Requirements are often a pain, between the one that you really use, and all the dependencies.
This repo advise you to use [Pip-tools](https://github.com/jazzband/pip-tools).

To do so, put your requirements in the requirements.in, then run 

 ` pip-compile requirements.in`

 This will generate automatically the requirements.txt with all the required dependencies.