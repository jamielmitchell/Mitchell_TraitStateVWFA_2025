# Mitchell_TraitStateVWFA_2025
Code associated with:
Mitchell, J. L., Yablonski, M., Stone, H. L., Fuentes-Jimenez, M., Takada, M. E., Tang, K. A., Tran, J. E., Chou, C., & Yeatman, J. D. (2025). Small or absent Visual Word Form Area is a trait of dyslexia. bioRxiv, 2025.01.14.632854. https://doi.org/10.1101/2025.01.14.632854

## General Note
The data associated with this manuscript and analysis can be found in a Stanford University Libraries Digital Repository at [https://doi.org/10.25740/bq006zp5312](https://doi.org/10.25740/bq006zp5312). In order for the code to run flawlessly, you should download this data and store it in a folder named `Source_Data` and then download the code and store it in a parent folder alongside `Source_Data`. `Source_Data` contains a csv file with data needed to create every table and figure of the manuscript. Each cv is named according to the display item it is used to generate. Below is a description of which display items can be generates with each notebook. 

## Location of Manuscript Display Items Code
### Python Code
#### The following lists the contents of each python notebook
- scores.ipynb: code pertaining to display item generation for assessment score data
    - Figure 3a
    - Table S13
- presence.ipynb: code pertaining to display item generation for ROI presence/detection data
    - Figure 2b
    - Figure 3b
- size.ipynb: code pertaining to display item generation for ROI size data
    - Figure 2c
    - Figure 2d
    - Figure 3c
    - Figure 4
    - Table S1
    - Table S2
    - Table S3 
    - Table S10
    - Figure S1
    - Figure S3
- activations.ipynb: code pertaining to display item generation for ROI activation/signal data
    - Figure 2e
    - Figure 5a
    - post-hoc analysis for VWFA-2 text and object activation changes (not a display item)
- selectivity.ipynb: code pertaining to display item generation for ROI text selectivity data
    - Figure 2f
    - Figure 5b
    - Table S5
- prediction.ipynb: code pertaining to display item generation for a prediction analysis of VWFA size with theoretically sufficient intervention dossage
    - Figure 6 (4 separate files to be combined into a single figure)
- participant_data.ipynb: code pertaining to display item generation for study timeline and fMRI motion quality data
    - Figure S4
    - Figure S5

### R Code
#### The following lists the contents of each r markdown file
- scores.Rmd: code pertaining to linear mixed effect model results for assessment score data
    - Table S6
    - Table S7
- presence.Rmd: code pertaining to generalized linear mixed effect model results for ROI presence/detection data
    - Table S8
    - Table S15
- size.Rmd: code pertaining to linear mixed effect model results for ROI size data
    - Table 1
    - Table 2
    - Table S9
    - Table S14
- adctivations.Rmd: code pertaining to linear mixed effect model results for ROI activation/signal data
    - Table S4
    - Table S11
    - Table S16
- selectivity.Rmd: code pertaining to linear mixed effect model results for ROI text selectivity data
    - Table S12
    - Table S17

## Other Files
- usefulFunctions.py - contains custom functions that are commonly used across notebooks
- code.Rproj - r project where the r notebooks live