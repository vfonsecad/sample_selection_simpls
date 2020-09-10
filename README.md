# sample selection
# by: valeria fonseca diaz valeria.fonsecadiaz@kuleuven.be
# this folder contains procedures to analyze unsupervised sample selection methods for simpls models

The focus of this project is to show an exhaustive search of the best sample selection techiques to build simpls models for chemometrics. as there are many factors involved into studying the effectiveness of a sample selection method (sample size, algorithm, input dimensionality, etc.) the number of models to fit to select the best subsample gets very high. With numba functions of simpls, it is possible to make an exhaustive search in a very short time.
The sample selection methods are based on methods by scikit-learn and methods in the R package https://cran.r-project.org/web/packages/prospectr

## folders:

- data: data cases. usually there is a data raw folder and a data prepared folder so data is structured to be ready for analysis.
- methodology: here there are the codes for simpls and sample selection. there is another procedure to read the data files as stored in the /data/~/data\_prepared folder
- experiments: a folder where compuational experiments are stored using the methods in /methodology.
- environment.yml: the conda environment information for reproduction of these analyses.





