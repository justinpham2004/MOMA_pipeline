import pandas as pd
import sys


def main():
    try:
        if len(sys.argv) != 4:
            raise ValueError("Error: Not enough arguments entered")
        #args should be: <merge_data.py> <script.csv> <script2.csv> <outputScript.csv>
        #Reading in system input and setting it to variables for readability
        args = sys.argv
        tid_file = args[1]
        ms_data_file = args[2]
        output_file = args[3]

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
            combined_file.to_csv(output_file, index = False)
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




if __name__ == "__main__":
    main()