#import pandas as pd
import sys


def main():


    ## Get input from cmd line
    if len(sys.argv) < 2:
        print("No clusters selected, try again!")
    else:
        print("Selected clusters are: ")
        for arg in sys.argv[1:]:
            
            print( "Cluster: ", arg)
            

    ##Read JSON file from input
    #df = pd.read_json()

    ##Get selected cluster data


    ## Create CSV and output CSV


if __name__ == "__main__":
    main()