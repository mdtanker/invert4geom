# How to contribute

## TLDR (Too long; didn't read)
* open a [GitHub Issue](https://github.com/mdtanker/invert4geom/issues/new/choose) describing what you want to do
* [fork](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) the main branch and clone it locally
* [create a branch](https://docs.github.com/en/get-started/using-github/github-flow#create-a-branch) for your edits
* make your changes, and commit them using the [Angular Commits Convention](https://www.conventionalcommits.org/en/v1.0.0-beta.4/#summary)
* [make a Pull Request](http://makeapullrequest.com/) for your branch

🎉 Thanks for considering contributing to this package! 🎉

<sub>Adapted from the great contribution guidelines of the [Fatiando a Terra](https://www.fatiando.org/) packages<sub>.

> This document contains some general guidelines to help with contributing to this code. Contributing to a package can be a daunting task, if you want help please reach out on the [GitHub discussions page](https://github.com/mdtanker/invert4geom/discussions)!

Any kind of help would be much appreciated. Here are a few ways to contribute:
* 🐛 Submitting bug reports and feature requests
* 📝 Writing tutorials or examples
* 🔍 Fixing typos and improving to the documentation
* 💡 Writing code for everyone to use

If you get stuck at any point you can create an issue on GitHub (look for the Issues tab in the repository).

For more information on contributing to open source projects, [GitHub's own guide](https://guides.github.com/activities/contributing-to-open-source/) is a great starting point if you are new to version control.
Also, checkout the [Zen of Scientific Software Maintenance](https://jrleeman.github.io/ScientificSoftwareMaintenance/) for some guiding principles on how to create high quality scientific software contributions.

## Contents

* [Quick Summary](#quick-summary)
* [What Can I Do?](#what-can-i-do)
* [Reporting a Bug](#reporting-a-bug)
* [Editing the Documentation](#editing-the-documentation)
* [Contributing Code](#contributing-code)
  - [Setting up Make](#setting-up-make)
  - [Setting up your environment](#setting-up-your-environment)
  - [Code style and linting](#code-style-and-linting)
  - [Testing your code](#testing-your-code)
  - [Documentation](#documentation)
  - [Committing changes](#committing-changes)
  - [Code review](#code-review)
* [Publish a new release](#publish-a-new-release)
* [Update the Dependencies](#update-the-dependencies)
* [Set up Binder](#set-up-the-binder-configuration)
* [Release Checklist](#release-checklist)

## Quick Summary
### Forking
```
    git clone https://github.com/mdtanker/invert4geom.git

    cd invert4geom
```
### Installing
```
    make create

    conda activate invert4geom

    make install
```
### Testing
```
    make test
```
or
```
    pytest tests/test_<MODULE>.py::<FUNCTION>
```
### Formatting
```
    make style
```
or
```
    make check
    make pylint
```

### Building docs
You can build the docs using:
```bash
    nox -s docs
```

or if you don't want them to automatically update
```bash
    nox -s docs --non-interactive
```

## What Can I Do?

* Tackle any issue that you wish! Some issues are labeled as **"good first issues"** to indicate that they are beginner friendly, meaning that they don't require extensive   knowledge of the project.
* Make a tutorial or example of how to do something.
* Provide feedback about how we can improve the project or about your particular use case.
* Contribute code you already have. It doesn't need to be perfect! We will help you clean things up, test it, etc.

## Reporting a Bug

Find the *Issues* tab on the top of the GitHub repository and click *New Issue*.
You'll be prompted to choose between different types of issue, like bug reports and feature requests.
Choose the one that best matches your need.
The Issue will be populated with one of our templates.
**Please try to fillout the template with as much detail as you can**.
Remember: the more information we have, the easier it will be for us to solve your problem.

## Editing the Documentation

If you're browsing the documentation and notice a typo or something that could be improved, please consider letting us know by [creating an issue](#reporting-a-bug) or submitting a fix (even better 🌟).
You can submit fixes to the documentation pages completely online without having to download and install anything:

* On each documentation page, there should be a " ✏️ Suggest edit" link at the very top (click on the GitHub logo).
* Click on that link to open the respective source file on GitHub for editing online (you'll need a GitHub account).
* Make your desired changes.
* When you're done, scroll to the bottom of the page.
* Fill out the two fields under "Commit changes": the first is a short title describing your fixes, start this with `docs: `, then your short title; the second is a more detailed description of the changes. Try to be as detailed as possible and describe *why* you changed something.
* Click on the "Commit changes" button to open a pull request (see below).
* We'll review your changes and then merge them in if everything is OK.
* Done 🎉🍺

Alternatively, you can make the changes offline to the files in the `docs` folder or the example scripts. See [Contributing Code](#contributing-code) for instructions.

# Contributing Code

**Is this your first contribution?**
Please take a look at these resources to learn about git and pull requests (don't hesitate to ask questions in the [GitHub discussions page](https://github.com/mdtanker/invert4geom/discussions)):

* [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/).
* Aaron Meurer's [tutorial on the git workflow](http://www.asmeurer.com/git-workflow/)
* [How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)

If you're new to working with git, GitHub, and the Unix Shell, we recommend starting with the [Software Carpentry](https://software-carpentry.org/) lessons, which are available in English and Spanish:

* 🇬🇧 [Version Control with Git](http://swcarpentry.github.io/git-novice/) / 🇪🇸 [Control de versiones con Git](https://swcarpentry.github.io/git-novice-es/)
* 🇬🇧 [The Unix Shell](http://swcarpentry.github.io/shell-novice/) / 🇪🇸 [La Terminal de Unix](https://swcarpentry.github.io/shell-novice-es/)

### General guidelines

We follow the [git pull request workflow](http://www.asmeurer.com/git-workflow/) to
make changes to our codebase.
Every change made to the codebase goes through a pull request so that our
[continuous integration](https://en.wikipedia.org/wiki/Continuous_integration) services
have a chance to check that the code is up to standards and passes all our tests.
This way, the *main* branch is always stable.

General guidelines for pull requests (PRs):

* **Open an issue first** describing what you want to do. If there is already an issue
  that matches your PR, leave a comment there instead to let us know what you plan to
  do.
* Each pull request should consist of a **small** and logical collection of changes.
* Larger changes should be broken down into smaller components and integrated
  separately.
* Bug fixes should be submitted in separate PRs.
* Describe what your PR changes and *why* this is a good thing. Be as specific as you
  can. The PR description is how we keep track of the changes made to the project over
  time.
* Write descriptive commit messages. Chris Beams has written a
  [guide](https://chris.beams.io/posts/git-commit/) on how to write good commit
  messages.
* Be willing to accept criticism and work on improving your code; we don't want to break
  other users' code, so care must be taken not to introduce bugs.
* Be aware that the pull request review process is not immediate, and is generally
  proportional to the size of the pull request.

### Setting up `make`

Most of the commands used in the development of `PolarToolkit` use the tool `make`.
The `make` commands are defined in the file [`Makefile`](https://github.com/mdtanker/polartoolkit/blob/main/Makefile), and are run in the terminal / command prompt with the format ```make <<command name>>```.

If you don't want to use `make`, you can always open the `Makefile` and copy and past the command you need into the terminal.

`make` is typically included in most unix systems, but can be installed explicitly with a package manager such as `Homebrew` for MacOS, or `RPM` or`DNF` for Linux, or `Chocalatey` for Windows.

Run `make -version` to test that `make` is correctly installed.

### Setting up your environment

To get the latest version clone the github repo:

```
git clone https://github.com/mdtanker/invert4geom.git
```
Change into the directory:

```
cd invert4geom
```

Run the following `make` command to create a new environment and install the package dependencies:

```
make create
```
Activate the environment:
```
conda activate invert4geom
```
Install your local version:
```
make install
```
This environment now contains your local, editable version of Invert4Geom, meaning if you alter code in the package, it will automatically include those changes in your environment (you may need to restart your kernel if using Jupyter).
If you need to update the dependencies, see the [update the dependencies](#update-the-dependencies) section below.

> **Note:** You'll need to activate the environment every time you start a new terminal.

### Code style and linting

We use [pre-commit](https://pre-commit.com/) to check code style. This can be used locally, by installing pre-commit, or can be used as a pre-commit hook, where it is automatically run by git for each commit to the repository. This pre-commit hook wont add or commit any changes, but will just inform your of what should be changed. Pre-commit is setup within the `.pre-commit-config.yaml` file. There are lots of hooks (processes) which run for each pre-commit call, including [Ruff](https://docs.astral.sh/ruff/) to format and lint the code. This allows you to not think about proper indentation, line length, or aligning your code during development. Before committing, or periodically while you code, run the following to automatically format your code:
```
make check
```

Go through the output of this and try to change the code based on the errors. Search the error codes on the [Ruff documentation](https://docs.astral.sh/ruff/), which should give suggestions. Re-run the check to see if you've fixed it. Somethings can't be resolved (unsplittable urls longer than the line length). For these, add `# noqa: []` at the end of the line and the check will ignore it. Inside the square brackets add the specific error code you want to ignore.

We also use [Pylint](https://pylint.readthedocs.io/en/latest/), which performs static-linting on the code. This checks the code and catches many common bugs and errors, without running any of the code. This check is slightly slower the the `Ruff` check. Run it with the following:
```
make pylint
```
Similar to using `Ruff`, go through the output of this, search the error codes on the [Pylint documentation](https://pylint.readthedocs.io/en/latest/) for help, and try and fix all the errors and warnings. If there are false-positives, or your confident you don't agree with the warning, add ` # pylint: disable=` at the end of the lines, with the warning code following the `=`.

To run both pre-commit and pylint together use:
```
make style
```

### Docstrings

**All docstrings** should follow the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
All functions/classes/methods should have docstrings with a full description of all arguments and return values.

While the maximum line length for code is automatically set by *Ruff*, docstrings must be formatted manually.
To play nicely with Jupyter and IPython, **keep docstrings limited to 88 characters** per line.
We don't have a good way of enforcing this automatically yet, so please do your best.

#### Type hints

We have also opted to use type hints throughout the codebase.
This means each function/class/method should be fulled typed, including the docstrings.
We use [mypy](https://mypy.readthedocs.io/en/stable/) as a type checker.
```
    make mypy
```
Try and address all the errors and warnings.
If there are complex types, just use `typing.Any`, or if necessary, ignore the line causing the issue by adding `# type: ignore[]` with the error code inside the square brackets.

### Testing your code

Automated testing helps ensure that our code is as free of bugs as it can be.
It also lets us know immediately if a change we make breaks any other part of the code.

All of our test code and data are stored in the `tests` subpackage.
We use the [pytest](https://pytest.org/) framework to run the test suite, and our continuous integration systems with GitHub Actions use CodeCov to display how much of our code is covered by the tests.

Please write tests for your code so that we can be sure that it won't break any of the existing functionality.
Tests also help us be confident that we won't break your code in the future.

If you're **new to testing**, see existing test files for examples of things to do.
**Don't let the tests keep you from submitting your contribution!**
If you're not sure how to do this or are having trouble, submit your pull request anyway.
We will help you create the tests and sort out any kind of problem during code review.

Run the tests and calculate test coverage using:

    make test

To run a specific test by name:

    pytest --cov=. -k "test_name"

The coverage report will let you know which lines of code are touched by the tests.
**Strive to get 100% coverage for the lines you changed.**
It's OK if you can't or don't know how to test something.
Leave a comment in the PR and we'll help you out.

### Documentation

The Docs are build with `Sphinx` and `Read the Docs`.
Due to the above mentioned issues with the included C programs, `Read the Docs (RTD)` can't run the scripts which are part of the docs (i.e. the gallery examples).
Because of this the notebooks don't execute on a build, as specified by `execute_notebooks: 'off'` in `_config.yml`.
Here is how to run/update the docs on your local machine.

> **Note:** The docs are automatically built on PR's by `RTD`, but it's good practice to build them manually before a PR, to check them for errors.

#### Run all .ipynb's to update them

    make run_docs

If your edits haven't changed any part of the core package, then there is no need to re-run the notebooks.
If you changed a notebook, just clear it's contents and re-run that one notebook.

#### Check the build manually (optional)

You can build the docs using, but this will require pandoc to be install on your machine:

```bash
nox -s docs
```

#### Automatically build the docs

Add, commit, and push all changes to GitHub in a Pull Request, and `RTD` should automatically build the docs.
In each PR, you will see section of the checks for `RTD`. Click on this to preview the docs for the PR.
`RTD` uses the conda environment specified in `env/RTD_env.yml` when it's building.

### Committing changes

Once your have made your changes locally, you'll need to make a branch, commit the changes, and create a PR. We use the [Angular Commits Convention](https://www.conventionalcommits.org/en/v1.0.0-beta.4/#summary) for commit messages. This allows automatic creation of the Changelogs, and automatic determination of what the next version will be. All commits should follow the below structure:
```
<type>: <description>

[optional body]
```

Where `type` is one of the following:
   * `docs` --> a change to the documents
   * `style`--> simple fixes to the styling of the code
   * `feat` --> any new features
   * `fix` --> bug fixes
   * `build` --> changes to the package build process, i.e. dependencies, changelogs etc.
   * `chore` --> maintenance changes, like GitHub Action workflows
   * `refactor` --> refactoring of the code without user-seen changes

The `optional body` can include any detailed description, and can optionally start with `BREAKING CHANGE: `.

Based on the commit message, one of four things will happen when;
1) no new version will be released
2) a `PATCH` version will be released (`v1.1.0 -> v1.1.1`)
3) a `MINOR` version will be released (`v1.1.0 -> v1.2.0`)
4) a `MAJOR` version will be released (`v1.1.0 -> v2.0.0`)

This follows [Semantic Versioning](https://semver.org/#summary) where given a version number `MAJOR.MINOR.PATCH`, the software should increment the:
1) `MAJOR` version when you make incompatible API changes
2) `MINOR` version when you add functionality in a backward compatible manner
3) `PATCH` version when you make backward compatible bug fixes

The following will occur based on your commit message:
* `BREAKING CHANGE: ` will always result in a `MAJOR` release
* `feat` will always result in a `MINOR` release
* `fix` will always result in a `PATCH` release

### Code Review

After you've submitted a pull request, you should expect to hear at least a comment within a couple of days.
We may suggest some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted quickly:

* Write a good and detailed description of what the PR does.
* Write tests for the code you wrote/modified.
* Readable code is better than clever code (even with comments).
* Write documentation for your code (docstrings) and leave comments explaining the *reason* behind non-obvious things.
* Include an example of new features in the gallery or tutorials.
* Follow the [numpy guide](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) for documentation.
* Run the automatic code formatter and style checks.

If you're PR involves changing the package dependencies, see the below instructions for [updating the dependencies](#update-the-dependencies).

Pull requests will automatically have tests run by GitHub Actions.
This includes running both the unit tests as well as code linters.
GitHub will show the status of these checks on the pull request.
Try to get them all passing (green).
If you have any trouble, leave a comment in the PR or [post on the GH discussions page](https://github.com/mdtanker/invert4geom/discussions).

## Publish a new release

This will almost always be done by the developers, but as a guide for them, here are instructions on how to release a new version of the package.
Follow all the above instructions for formatting. Push your changes to a new or existing Pull Request.
Once the automated GitHub Actions run (and pass), merge the PR into the main branch.

### PyPI (pip)
PyPI release are made automatically via GitHub actions whenever a pull request is merged.

### Conda-Forge
Once the new version is on PyPI, within a few hours a bot will automatically open a new PR in the [Invert4Geom conda-forge feedstock](https://github.com/conda-forge/invert4geom-feedstock).
Go through the checklist on the PR.
Most of the time the only actions needs are updated any changes made to the dependencies since the last release.
Merge the PR and the new version will be available on conda-forge shortly.

Once the new version is on conda, update the binder .yml file, as below.

## Update the dependencies

To add or update a dependencies, add it to `pyproject.toml` either under `dependencies` or `optional-dependencies`.
This will be included in the next build uploaded to PyPI.

If you add a dependency necessary for using the package, make sure to add it to the Binder config file.
See below.

## Set up the binder configuration

To run this package online, Read the Docs will automatically create a Binder instance based on the configuration file `binder/environment.yml`.
This file should reflect the latest release on Conda-Forge.
To allow all optional features in Binder, we need to manually add optional dependencies to the `binder/environment.yml` file.

Now, when submitting a PR, `RTD` will automatically build the docs and update the Binder environment.

## Release Checklist
* re-run any relevant notebooks
* check docs are building correctly using the GitHub actions link within the PR
* merge the PR
* wait for `PyPI` to publish the new version [here](https://pypi.python.org/pypi/invert4geom)
* wait for a PR to be opened in the [feedstock](https://github.com/conda-forge/invert4geom-feedstock)
* update any changed dependencies in the feedstock PR and merge
* wait for `conda` to publish the new version [here](https://anaconda.org/conda-forge/invert4geom)
* manually add dependency changes to `environment.yml`
* update invert4geom version in `binder/environment.yml`.
* test `PyPI` version with:
    - `make remove`
    - `make create_env`
    - `mamba activate invert4geom`
    - `make pip_install`
    - `make test`
* test `conda` version with:
    - `make remove`
    - `make conda_install`
    - `mamba activate invert4geom`
    - `make test`