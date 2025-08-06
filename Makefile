PROJECT=invert4geom

####
####
# install commands
####
####

create:
	mamba env create --file environment.yml --name $(PROJECT)

install:
	pip install --no-deps -e .

update:
	mamba env update --file environment.yml --name $(PROJECT) --prune

remove:
	mamba env remove --name $(PROJECT)


####
####
# doc commands
####
####

clear_examples:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/examples/*.ipynb

run_examples:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=False --execute --inplace docs/examples/*.ipynb

clear_how_to:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/how_to/*.ipynb

run_how_to:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=False --execute --inplace docs/how_to/*.ipynb

clear_tutorial:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/tutorial/*.ipynb

run_tutorial:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=False --execute --inplace docs/tutorial/*.ipynb

clear_docs: clear_examples clear_how_to clear_tutorial
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/*.ipynb

run_docs: run_examples run_how_to run_tutorial
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*.ipynb
