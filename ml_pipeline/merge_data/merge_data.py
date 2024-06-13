import pandas as pd
import sys
import csv

def main():
    #args should be: <merge_data.py> <script.csv> <script2.csv> <outputScript.csv>
    args = sys.argv
    tid_file = args[1]
    ms_data_file = args[2]
    output_file = args[3]

    input_tids = pd.read_csv(tid_file)
    ms_data = pd.read_csv(ms_data_file)

    combined_file = input_tids.merge(ms_data, how = 'left', on = 'USID')
    
    combined_file.to_csv(output_file, index = False)



if __name__ == "__main__":
    main()