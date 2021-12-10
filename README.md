# pfactor_gradients

# Environment build (Macbook)

    conda create -n pfactor_gradients python=3.7
    conda activate pfactor_gradients

    # Essentials
    pip install ipython pandas numpy seaborn matplotlib nibabel nilearn ipywidgets tqdm

	# Tertiary
	pip install scipy statsmodels sklearn pingouin brainspace bctpy shap pygam abagen networkx
	conda install -c conda-forge control

    # Octave support
    pip install oct2py

    cd /Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients
    conda env export > environment.yml --no-builds


# Environment build (cubic, home)

    conda create -n pfactor_gradients python=3.7
    conda activate pfactor_gradients

    # Essentials
    pip install ipython pandas numpy seaborn matplotlib nibabel nilearn ipywidgets tqdm

    # Tertiary
	pip install scipy statsmodels sklearn pingouin brainspace bctpy shap pygam abagen
	conda install -c conda-forge control

    # Octave support
    pip install oct2py

    cd /cbica/home/parkesl/research_projects/pfactor_gradients/
    conda env export > environment_cluster.yml --no-builds
