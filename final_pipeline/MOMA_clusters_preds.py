import pandas as pd
import sys
import csv

import sys
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
    # Swap Key and Value of the labels so they can be mapped to the classification report.
    inverted_dict = {str(v): k for k, v in val_class_names.items()}
    inverted_dict['accuracy'] = 'Accuracy'
    inverted_dict['macro avg'] =  'Macro Average'
    inverted_dict['weighted avg']= 'Weighted Average'
    
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    
    df.insert(0, 'category_labels', inverted_dict)
    df = df.reset_index()
    df['category_labels'] = df['index'].map(inverted_dict)
    df.to_csv(output_name)
    

"""
    encode_labels(label_type, df)

    Input:
        label_type = what is being encoded, ie; category label, sample label, etc.
        df = original dataframe being loaded
    Output:
        returns (list of label_names, list of encoded labels)

"""
def encode_labels(label_type, df, test_df): 
    ## Variables needed for label mapping ##    
    test_labels = None
    labels = None
    test_label_map = None
   
    if label_type == 'sample':
        test_labels = test_df['sample_label'] 
        labels = df['sample_label']
    elif label_type == 'category':
        test_labels = test_df['category_label'] ##These are just the full columns of data from the NN
        labels = df['category_label']
    else:
        raise ValueError("Enter a Valid Label Type!")

    test_label_map = getLabelEncodedy(test_labels)[1] 

    matched_labels = dict()
    unmatched_labels = set()
    for label in labels:
        if label in test_label_map:
            matched_labels[label] = test_label_map[label]
        else:
            matched_labels[label] = 999

    encoded_set = [matched_labels[label] for label in labels]

    return (test_label_map, encoded_set)


def get_label(neural_network):
    split = neural_network.split("_")
    for word in split:
        word = word.lower()
        if word == "sample" or word == "category":
            return word
    raise ValueError("Label Type Not Found: Specify what label is being searched for in neural network name. Ex: 'labeltype_neural_network'")


def get_clusters(cluster, file_path):
     try:
            f = pd.read_json(file_path)
            #print(f.head())
     except:
            raise ValueError("Error! JSON file not found")

    

     with open(f"cluster_{cluster}.csv", 'w') as file:
            csvwriter = csv.writer(file)
            column_labels = ['USID', 'cluster_algo', 'cluster_number']
            csvwriter.writerow(column_labels)

            try:
                ## Data to be written into CSV ##
                index = int(cluster)
                cluster_algo = f['clustering_algorithm'].at[index]
                tid_list = f['clusters'].at[index]
                test_ids = tid_list['scanIDs']
                
                #print("Cluster", index, "is being written to the csv right now")
                
                for tid in test_ids:
                    row_entry = [tid, cluster_algo, cluster]
                    csvwriter.writerow(row_entry)

            except (KeyError, IndexError) as e:
                print(f"Error Processing cluster {index}: {e} is not a valid cluster")
            except ValueError:
                print(f"Invalid cluster index: {index}")


def merge_data(cluster, ms_data_file):
    tid_file = f"cluster_{cluster}.csv"
    try:
        #Reading data from files
        try:
            input_tids = pd.read_csv(tid_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {tid_file}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"File is empty: {tid_file}")
        except pd.errors.ParserError:
            raise ValueError(f"Parsing error in file: {tid_file}")
    
        #reading mass-spectrometer data from file
        try:
            ms_data = pd.read_csv(ms_data_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {ms_data_file}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"File is empty: {ms_data_file}")
        except pd.errors.ParserError:
            raise ValueError(f"Parsing error in file: {ms_data_file}")

        #merging files with pandas merge
        try:
            combined_file = input_tids.merge(ms_data, how = 'left', on = 'USID')
        except KeyError as e:
            raise KeyError(f"Key error during merge file: {e}")
        
        #outputting merged file to given name
        try:
            combined_file.to_csv(f"cluster_{cluster}_data.csv", index = False)
        except IOError as e:
            raise IOError(f"IO error during writing file: {e}")
        
    #Other possible errors to be thrown
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except FileNotFoundError as fnfe:
        print(f"FileNotFoundError: {fnfe}")
    except KeyError as ke:
        print(f"KeyError: {ke}")
    except IOError as ioe:
        print(f"IOError: {ioe}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




def make_predictions(cluster, neural_network, output, training_data):    
    
    """
    Get label name from nn_name:
    """
    label_type = get_label(neural_network)
   
    """
    Testfile Data Housekeeping
    - Drop columns 1050-2000 as those are empty MOMA data. Note: Using df.loc() drops a slice of labels!
    - Drop mcalDefault: all data entrys just come out to 1
    """
    try:
        full_datafile = pd.read_csv(f"cluster_{cluster}_data.csv")
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    datafile = full_datafile.drop(full_datafile.loc[:,'1050':'2000'].columns, axis = 1)
    datafile.drop('mcalDefault', axis = 1, inplace = True)
    datafile.drop('cluster_number', axis = 1, inplace = True)

    try:
        test_datafile = pd.read_csv(training_data)
    except Exception as e:
        print(f"Error reading training data file: {e}")
        sys.exit(1)
    
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
    #print(mass_spec_data.head) #(For Testing purposes)#

    ## Step 3: Normalization of relevant data
    ms_normedMax = mass_spec_data.apply(lambda x: x/x.max(), axis = 1)

    ## Step 4: Label Encoding: By matching labels from the testing data used for these NN, the NN can correctly label the prediction to the correct label#   
    label_set, encoded_testset = encode_labels(label_type, full_datafile, test_datafile) 
    print(encoded_testset)

    ## NN for Sample Labels: Loading the model and predicting the data
    loaded_model = tf.keras.models.load_model(neural_network, compile = True)
    predictions_loaded_model = loaded_model.predict(ms_normedMax)
    predictions = np.argmax(predictions_loaded_model, axis = 1)

    ## Generate Output
    generate_classification_report_test(encoded_testset, predictions, label_set, output)
    
    ## These are sample data to be used for testing ##
    #generate_classification_report_test(category_encoded_testset, predictions, category_labels, "category_label_test.csv")
    #generate_classification_report_test(sample_encoded_testset, predictions, sample_labels, "sample_label_test.csv")





def main():
    TRAINING_DATA = './data/modified_updated_alldata2023-TIC30filtered.csv' #alter this because it depends on the nn model

    # CMD Line arguments that will be used to feed into methods
    args = sys.argv
    if len(sys.argv) != 6:
            raise ValueError("Incorrect arguments provided. Usage: MOMA_clusters_preds.py <cluster #> <cluster_file>.json <input_data>.csv <neuralnetwork>.h5 <output_name>.csv")

    cluster_name = args[1]
    cluster_file = args[2]
    ms_data_file = args[3]
    neural_network = args[4]
    output_file = args[5]

    ## Get clusters from file, and create .csv of just testids and metadata of tests from the selected clusters.
    get_clusters(cluster_name, cluster_file)

    ## Merge Test IDS with the Mass Spectrometer Data associated with it. Output a .csv file of merged data for future testing
    merge_data(cluster_name, ms_data_file)

    ## Make Predictions and generate prediction report.
    make_predictions(cluster_name, neural_network, output_file, TRAINING_DATA)


if __name__ == "__main__":
    main()
    