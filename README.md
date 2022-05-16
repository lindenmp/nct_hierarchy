# nct_hierarchy

The goal of this project was to examine whether the topology of the structural connectome—which we index using an
undirected description of connectivty—confers asymmetries in signal propagation across the cortical hierarchy of 
cytoarchitecture. We use Network Control Theory to model state transitions that span the sensory-fugal axis of
cytoarchitectonic similarity and examine whether and how dynamics differed for bottom-up state transitions compared to
top-down.

The preprint for this work can be found here:

Asymmetric Signaling Across the Hierarchy of Cytoarchitecture within the Human Connectome. Linden Parkes, Jason Z Kim, Jennifer Stiso, Monica E Calkins, Matthew Cieslak, Raquel E Gur, Ruben C Gur, Tyler M Moore, Mathieu Ouellet, David R Roalf, Russell T Shinohara, Daniel H Wolf, Theodore D Satterthwaite, Dani S Bassett
bioRxiv 2022.05.13.491642; doi: https://doi.org/10.1101/2022.05.13.491642

Also see the following YouTube video for a short animation describing this work: https://www.youtube.com/watch?v=cbnS6WamXzE

## scripts overview

Python scripts that produce the figures from the above manuscript can be found in `scripts`. Many of these figure 
scripts are supported by additional scripts that run on our cluster at Penn. Those scripts are stored here for documentation 
purposes. However, the main `results_figX.py` scripts were designed to run with or without the outputs 
from these cluster scripts. Of course, some of them will have much longer run times without the cluster outputs.