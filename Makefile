PROJECT=invert4geom
VERSION := $(shell grep -m 1 'version =' pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)

print-%  : ; @echo $* = $($*)
####
####
# install commands
####
####

create:
	mamba create --name $(PROJECT) --yes --force --channel conda-forge polartoolkit python=3.11

install:
	pip install -e .[all]

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

clear_gallery_outputs:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/gallery/*.ipynb

run_gallery: clear_gallery_outputs
	jupyter nbconvert --ExecutePreprocessor.allow_errors=False --execute --inplace docs/gallery/*.ipynb

clear_user_guide_outputs:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/user_guide/*.ipynb

run_user_guide: clear_user_guide_outputs
	jupyter nbconvert --ExecutePreprocessor.allow_errors=False --execute --inplace docs/user_guide/*.ipynb

clear_docs_outputs:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/**/*.ipynb

run_doc_files: clear_docs_outputs
	jupyter nbconvert --ExecutePreprocessor.allow_errors=False --execute --inplace docs/**/*.ipynb
