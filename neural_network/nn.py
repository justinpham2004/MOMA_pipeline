## What imports?
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.metrics import classification_report


"""ML Pipeline Goals

Write python script 3 that predicts chemical family and composition of the selected TIDs.
Inputs: csv file just created with MS values of selected TIDs, h5 file (saved NN model)
Output: csv file with TIDs and associated predictions

python  predict_selectedclusters.py  output_MSselectedclusters.csv   saved_NN.h5  predictions_selectedclusters.csv


Notes: this script will need to preprocess the input data the same way it was done for the NN algorithm script
"""

## Helper Functions:

"""
    getLabelEncoded(y): 
    
    Input: list of all unique labels within the datafile, 
    of sample and category. (Like names of elements, compositions, etc.)

    Output: Uses the label encoder to encode all labels into a series of numbers, 
    this series of numbers is put into a dictionary, with the category name 
    as the key, and the transformed number as the value.

"""
def getLabelEncodedy(y):
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y)   
    label_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return(encoded_y , label_map) #dicLabelsEncoded)

"""
    generate_classification_report_test(true_labels, predicted_labels, val_class_names)

    Input: 
        true_labels = correct label 
        predicted_label = neural network predicted label
        val_class_names = What type of label will the NN be predicting
    Output:
        Uses sk.learn to generate a classification report of the predicted data (after the nn is run on the test data.)
        Creates the output file (.csv) based on the user input
"""
def generate_classification_report_test(true_labels, predicted_labels, val_class_names, output_name):
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    df = pd.DataFrame(report).transpose()

    # Add class Names
    new_class_names = val_class_names[:]
    if len(val_class_names) < len(df.index):
        new_class_names.extend([''] * (len(df.index) - len(new_class_names)))
    df.insert(0, 'sample_labels', new_class_names)
    df = df.reset_index().rename(columns={'index': 'sample_labels'})
    df.to_csv(output_name)

"""
    encode_labels(label_type, df)

    Input:
        label_type = what is being encoded, ie; category label, sample label, etc.
        df = original dataframe being loaded
    Output:
        returns (list of label_names, list of encoded labels)

"""
def encode_labels(label_type, df): 
    labels = df[f'{label_type}_label']

    encoded_labels, labelmap = getLabelEncodedy(labels)
    label_names = list(labelmap)
    encoded_testset = [labelmap[label] for label in labels]
    return (label_names, encoded_testset)

def get_label(neural_network):
    split = neural_network.split("_")
    for word in split:
        word = word.lower()
        if word == "sample" or word == "category":
            return word
    raise ValueError("Label Type Not Found: Specify what label is being searched for in neural network name. Ex: 'labeltype_neural_network'")

""" Command Line Arguments:
    # 1: script name
    # 2: CSV with TIDS and metadata
    # 3: saved NN model (name)
    # 4: Predicted clusters csv
    """
def main():
    """
    if len(sys.argv) != 4:
        raise ValueError("Use Format: python  nn.py  <input_clusters>.csv  <neural_network_name.h5> <nn_predictions>.csv")
    """
    #args = sys.argv[1:]
    #input = args[0]
    #neural_network = args[1]
    #output = args[2]
    input = 'data/nn_data.csv'
    neural_network = '/models/xinadata2021v2-NoTIC30_Model1Dap2022_category_7'
    output = 'output/practice.csv'

    
    """
    Get label name from nn_name:
    """
    label_type = get_label(neural_network)
    
   
    """
    Testfile Data Housekeeping
    - Drop columns 1050-2000 as those are empty MOMA data. Note: Using df.loc() drops a slice of labels!
    - Drop mcalDefault: all data entrys just come out to 1
    """
    full_datafile = pd.read_csv(input)
    datafile = full_datafile.drop(full_datafile.loc[:,'1050':'2000'].columns, axis = 1)
    datafile.drop('mcalDefault', axis = 1, inplace = True)


    """
    Data Preprocessing Step:
    1. Read in massspec data as a df.slice
    2. Separate MetaData and MassSpec in pd
        #print(mass_spec_data) 
        #print(mass_spec_params) 
        #This data consists of all the sample labels that are number based parameters
        #Debugging Purposes
    3. Normalize mass spec data
    - Standardize Data
    4: Label Encode: Categories/Sample (since they are unique elements, important text to keep)
    - Split data: Test set and Training set
    """
    ##Step 1, 2: Separating Meta Data and Mass Spec
    mass_spec_data = datafile.iloc[:,3:-10] ## Columns 3 to -10, only data
    #mass_spec_params = datafile.iloc[:,-10:-3] ## All columns up to -10 # In case you want parameter data

    ## Step 3: Normalization of relevant data
    ms_normedMax = mass_spec_data.apply(lambda x: x/x.max(), axis = 1)
    #params_normedMax = mass_spec_params.apply(lambda x: x/x.max(), axis = 1) ## Isnt really used in Victorias Code?

    ## Step 4: Label Encoding
    label_names, encoded_testset = encode_labels(label_type, full_datafile)

    ## NN for Sample Labels:
    loaded_model = tf.keras.models.load_model(neural_network, compile = True)
    predictions_loaded_model = loaded_model.predict(ms_normedMax)
    predictions = np.argmax(predictions_loaded_model, axis = 1)
    
    ## Generate Output
    generate_classification_report_test(encoded_testset, predictions, label_names, label_type, output)



if __name__ == "__main__":
    main()
    