import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz


#print(fuzz.ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room'))

micro_precision = 0
macro_precision = 0
micro_recall = 0
macro_recall = 0

predictions = [(str(v[0]).lower(),str(v[1]).lower()) for v in pd.read_csv("spacy_extended.csv",header=None).values]
actuals = [(str(v[0]).lower(),str(v[1]).lower()) for v in pd.read_csv("test_dataset.csv",header=None).values]

precision = len(set(predictions).intersection(set(actuals))) / len(set(predictions))
recall = len(set(predictions).intersection(set(actuals))) / len(set(actuals))
print("Precision= "+str(precision))
print("Recall= "+str(recall))

intersections = 0

for p in predictions:
    for a in actuals:
        p = list(p)
        a = list(a)
        try:
           if fuzz.ratio(a[0],p[0])>80 and fuzz.ratio(a[1],p[1])>80:
              intersections = intersections + 1
              break
        except:
             continue

print("Micro precision = "+str(intersections/len(predictions)))
print("Micro recall = "+str(intersections/len(actuals)))