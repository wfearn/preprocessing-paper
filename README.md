# Introduction
A repository with the code used to produce the results in the NAACL2021 paper titled Exploring the Relationship Between Algorithm Performance, Vocabulary, and Run-Time in Text Classification.

# Information
Here is a list of the directories and what they contain:

preprocess - the main code folder, can be accessed as a package in python by running 'python3 setup.py --install'

scripts - Folder with bash and slurm scripts that I used to get these results on BYUs supercomputer.

test - Folder with unit tests that I created to make sure the code in preprocess ran correctly.
You can run these with pytest.

utilities - Folder meant to hold miscellaneous files important to code execution, currently just holds
an English stopword list.

# Setup
Ensure datasets are available in the $HOME/.preprocess/downloads folder.
