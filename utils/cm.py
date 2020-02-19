import pandas as pd
import numpy as np

df = pd.read_csv("ph.csv")
OUTPATH = "/cluster/home/jessesun/diagnosis.txt"

with open(OUTPATH, "a") as o:
	for idx, row in df.iterrows():
		o.write(str(row["Study ID"]) + " " +  str(row["cadiotoxicity_yes_no"]) + "\n")
