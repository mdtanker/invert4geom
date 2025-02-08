PROJECT=invert4geom
VERSION := $(shell grep -m 1 'version =' pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)

print-%  : ; @echo $* = $($*)
####
####
# install commands
####
####

create:
	mamba env create --file environment.yml

install:
	pip install --no-deps -e .

remove:
	mamba remove --name $(PROJECT) --all

pip_install:
	pip install $(PROJECT)[all]==$(VERSION)

conda_install:
	mamba create --name $(PROJECT) --yes --force --channel conda-forge $(PROJECT)=$(VERSION) pytest pytest-cov ipykernel

####
####
# test commands
####
####

test: test_coverage test_numba

test_coverage:
	NUMBA_DISABLE_JIT=1 pytest

test_numba:
	NUMBA_DISABLE_JIT=0 pytest -rP -m use_numba

####
####
# style commands
####
####

check:
	pre-commit run --all-files

pylint:
	pylint $(PROJECT)

style: check pylint

mypy:
	mypy src/$(PROJECT)

####
####
# chore commands
####
####

release_check:
	semantic-release --noop version

changelog:
	semantic-release changelog

clean:
	find . -name '*.pickle' -delete
	find . -name '*.log' -delete
	find . -name '*.lock' -delete
	find . -name '*.pkl' -delete
	find . -name '*.sqlite3' -delete
	find . -name '*.coverage' -delete

####
####
# doc commands
####
####

clear_examples:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/examples/*.ipynb

run_examples: clear_examples
	jupyter nbconvert --ExecutePreprocessor.allow_errors=False --execute --inplace docs/examples/*.ipynb

clear_how_to:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/how_to/*.ipynb

run_how_to: clear_how_to
	jupyter nbconvert --ExecutePreprocessor.allow_errors=False --execute --inplace docs/how_to/*.ipynb

clear_tutorial:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/tutorial/*.ipynb

run_tutorial: clear_tutorial
	jupyter nbconvert --ExecutePreprocessor.allow_errors=False --execute --inplace docs/tutorial/*.ipynb

clear_docs:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/**/*.ipynb

run_docs: clear_docs
	jupyter nbconvert --ExecutePreprocessor.allow_errors=False --execute --inplace docs/**/*.ipynb
