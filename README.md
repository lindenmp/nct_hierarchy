# pfactor_gradients

# Environment build

    conda create -n pfactor_gradients python=3.7
    conda activate pfactor_gradients

    # Essentials
    pip install jupyterlab ipython pandas numpy seaborn matplotlib nibabel nilearn ipywidgets tqdm
    pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install

	# Statistics
	pip install scipy statsmodels sklearn pingouin brainspace bctpy
	conda install -c conda-forge control

    cd /Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients
    conda env export > environment.yml
	pip freeze > requirements.txt
