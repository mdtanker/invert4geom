PROJECT=invert4geom
STYLE_CHECK_FILES=.

####
####
# install commands
####
####

create:
	mamba create --name $(PROJECT) --yes --force polartoolkit python=3.11

install:
	pip install -e .[all]

install_test:
	pip install $(PROJECT)[all]

remove:
	mamba remove --name $(PROJECT) --all

conda_install:
	mamba create --name $(PROJECT) --yes --force $(PROJECT)

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

format:
	ruff format $(STYLE_CHECK_FILES)

check:
	ruff check --fix $(STYLE_CHECK_FILES)

lint:
	pre-commit run --all-files

pylint:
	pylint $(PROJECT)

style: format check lint pylint

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


####
####
# doc commands
####
####

run_gallery:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/gallery/*.ipynb

run_user_guide:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/user_guide/*.ipynb

run_all_doc_files: run_gallery run_user_guide
