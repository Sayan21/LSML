We included the codes, but did not include the data due to the lack of space. Only small datasets like Bibtex and Delicious datasets have been included. The rest of the datasets are available at http://research.microsoft.com/en-us/um/people/manik/downloads/XC/XMLRepository.html

The codes will require Tensor Toolbox version 2.5 from Sandia National Library. Please do not use any other version.

Once the training is completed, the AUC is computed using the script MAP_MultiLabel.py. Please call the script MAP_MultiLabel.py as,
python MAP_MultiLabel.py <docfile>  <labelfile> <testfile (in SVM format)>
