import pandas as pd
import sys
import csv


def main():

    try:
        #User Input Arguments
        args = sys.argv

        if len(sys.argv) < 3:
            raise ValueError("Not enough arguments provided. Usage: get_clusters.py <file_path>.json <cluster_names> ... <output_file>.csv")


        file_path = args[1]
        cluster_names = args[2:-1]
        output_file = args[-1]
        

        ## Get input from cmd line
        ## Catch errors with input here. use try catch and end program if it is not correct
       
        ##Read JSON file from input, remember to sanitize input once finished##
        try:
            f = pd.read_json(file_path)
            #print(f.head())
        except:
            raise ValueError("Error! JSON file not found")

    

        with open(output_file, 'w') as file:
            csvwriter = csv.writer(file)
            column_labels = ['USID', 'cluster_algo', 'cluster_number']
            csvwriter.writerow(column_labels)

            for index in cluster_names:
                try:
                    ## Data to be written into CSV ##
                    index = int(index)
                    cluster = index
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
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"an unexpected error occured: {e}")


if __name__ == "__main__":
    main()