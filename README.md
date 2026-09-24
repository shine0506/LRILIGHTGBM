# LRILightGBM

Supplementary R code associated with the development and external
validation of a LightGBM model for predicting late recurrent
intussusception in children.

## Code

`Code_for_Model_Development_and_Validation.R`

The R script provides the analytical workflow for data preprocessing,
feature selection, machine-learning model development, internal and
external validation, SHAP interpretation, LightGBM feature-importance
assessment, and Shiny deployment.

## Data privacy

No patient-level clinical data are included in this repository.
Predictor names in the publicly available code are represented by
generic placeholders to protect patient privacy and comply with
institutional ethical and data-protection requirements.

## Software

R version 4.4.1.

The R script also includes the procedures used to extract and export LightGBM feature-importance measures, including normalized Gain and Frequency weights.
