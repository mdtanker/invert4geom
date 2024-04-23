PROJECT=invert4geom
STYLE_CHECK_FILES=.

create:
	mamba create --name $(PROJECT) --yes --force polartoolkit python=3.11

create_test_env:
	mamba create --name test --yes python=3.11

install:
	pip install -e .[all]

install_test:
	pip install invert4geom[all]

remove:
	mamba remove --name $(PROJECT) --all

test: test_coverage test_numba

test_coverage:
	NUMBA_DISABLE_JIT=1 pytest

test_numba:
	NUMBA_DISABLE_JIT=0 pytest -rP -m use_numba

format:
	ruff format $(STYLE_CHECK_FILES)

check:
	ruff check --fix $(STYLE_CHECK_FILES)

lint:
	pre-commit run --all-files

pylint:
	pylint invert4geom

style: format check lint pylint

mypy:
	mypy src/invert4geom

release_check:
	semantic-release --noop version

changelog:
	semantic-release changelog

run_gallery:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/gallery/*.ipynb

run_user_guide:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/user_guide/*.ipynb

run_all_doc_files: run_gallery run_user_guide

# install with conda
conda_install:
	mamba create --name invert4geom --yes --force invert4geom

# create binder yml
binder_env:
	mamba env export --name invert4geom --no-builds > binder/environment.yml
	# delete last line
	sed -i '$$d' binder/environment.yml
	# add pip and optional packages
	sed -i '$$a\  - pip\n  - pip:' binder/environment.yml
	sed -i '$$a\    - optuna>=3.1.0' binder/environment.yml
	sed -i '$$a\    - botorch>=0.4.0' binder/environment.yml
	sed -i '$$a\    - joblib' binder/environment.yml
	sed -i '$$a\    - psutil' binder/environment.yml
	sed -i '$$a\    - tqdm_joblib' binder/environment.yml
	sed -i '$$a\    - pyvista' binder/environment.yml
	sed -i '$$a\    - trame' binder/environment.yml
	sed -i '$$a\    - ipywidgets' binder/environment.yml
	sed -i '$$a\    - matplotlib' binder/environment.yml
	sed -i '$$a\    - seaborn' binder/environment.yml
	sed -i '$$a\    - ipython' binder/environment.yml
	# sed -i '$$a\    - ' binder/environment.yml

# create ReadTheDocs yml
RTD_env:
	mamba remove --name RTD_env --all --yes
	mamba create --name RTD_env --yes --force python==3.11 polartoolkit
	mamba env export --name RTD_env --no-builds --from-history > env/RTD_env.yml
	# delete last line
	sed -i '$$d' env/RTD_env.yml
	# add pip and install local package
	sed -i '$$a\  - pip\n  - pip:\n    - ../.[docs]' env/RTD_env.yml

# create testing yml
testing_env:
	mamba remove --name testing_env --all --yes
	mamba create --name testing_env --yes --force polartoolkit
	mamba env export --name testing_env --no-builds --from-history > env/testing_env.yml
	# delete last line
	sed -i '$$d' env/testing_env.yml
	# add pip and install local package
	sed -i '$$a\  - pip\n  - pip:\n    - ../.[test]' env/testing_env.yml
