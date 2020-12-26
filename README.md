# results_interpretation_by_shap
## Interpretation results of machine learning models by SHAP
The problem of binary classification of breast cancer in women was chosen as an example

To get an overview of which features are most important for a model we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. 
![Screenshot](summary_plot.jpeg)

The explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue
![Screenshot](screen_of_single_plot_html.jpeg)
