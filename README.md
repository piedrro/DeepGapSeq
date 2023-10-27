# DeepGapSeq

A **developmental** smFRET analysis package, including deeplearning for gap state identification. 
Includes embedded a modified version ebFRET (MATLAB). 
Deep Learning and trace generation modules Inspired by DeepLASI and DeepFRET.

## Installing DeepGapSeq From GitHub

    conda create –-name DeepGapSeq python==3.9
    conda activate DeepGapSeq
    conda install -c anaconda git
    conda update --all

    pip install git+https://github.com/piedrro/DeepGapSeq.git

## To install **MATLAB** engine (Windows):

python 3.9 requires MATLAB >= 2021b

MATLAB compatibility: https://uk.mathworks.com/support/requirements/python-compatibility.html

    pip install matlabengine

This will likely fail due to a MATLAB version issue. 
Read the traceback, and install the recomended verison. 
Then try again: 'pip install matlabengine==XXXX'