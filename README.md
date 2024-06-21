MOMA Pipeline Project. 

To use get_clusters.py

python get_clusters.py <cluster_data.json> <1> .... <cluster no.#> <output.csv>
cluster data: name of json file with data.
output: desired name of output file.

To use merge_data.py

The args should be: <merge_data.py> <script.csv> <script2.csv> <outputScript.csv>
script.csv: TID's and clusters that will get their MOMA data
script2.csv: MOMA data that will be matched with script.csv
outputScript.csv: merged data output.

Note: if there is no MOMA data to match with a given TID in script.csv, result will be NAN for those entries

