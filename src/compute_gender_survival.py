import csv
import numpy
import io

with io.open("data/train.csv", "rb") as csv_in:
    csv_data = csv.reader(csv_in)
    print csv_data.next()
    
    data = []
    for row in csv_data:
        data.append(row)

data = numpy.array(data)
survival = data[..., 1].astype(numpy.float)
print "Overall survival rate", survival.mean()

female_mask = data[...,4] == "female"

print "Female survival rate", survival[female_mask].mean()
print "Male survival rate", survival[numpy.logical_not(female_mask)].mean()

