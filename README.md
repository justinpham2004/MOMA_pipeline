MOMA Pipeline Project:
Finished Product is in final_pipeline folder. The three other folders represent the 3 scripts that were written and merged to form the final script.

to use: MOMA_clusters_preds.py, here is the documentation.
MOMA_clusters_preds.py <cluster #> <cluster_file>.json <input_data>.csv <neuralnetwork>.h5 <output_name>.csv

an example I have been using:

MOMA_clusters_preds.py <cluster #> test_sample.json test_sample1.csv modified_updated_alldata2023-TIC30filtered_Model1Dap2022_sample_7.h5 cluster_{#}_sample_predictions.csv

Note: Training model must contain the label type within the name, separated from other chars using _ _. ex "blablabla_sample_bla". Also, this pipeline is only modeled to look at category or sample data, so if wanted to addapt to other "label_types", update the encode_labels function to include the new lable.

Also note: if you reuse the same output file name as the last time it was run then new data won't be overwritten, it simply will not be output.
