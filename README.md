# DeepGapSeq

## Installing DeepGapSeq From GitHub

    conda create –-name DeepGapSeq python==3.9
    conda activate DeepGapSeq
    conda install -c anaconda git
    conda update --all

    pip install https://github.com/piedrro/DeepGapSeq.git

### To install **MATLAB** engine (Windows):

    pip install matlabengine

    This will likely fail due to a MATLAB version issue. 
    Read the traceback, and install the recomended verison. 
    Then try again: 'pip install matlabengine==XXXX'