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

fare_ceiling = 40
data[data[..., 9].astype(numpy.float) >= fare_ceiling, 9] = fare_ceiling - 1.
fare_bracket_size = 10

num_brackets = fare_ceiling / fare_bracket_size

num_classes = 3

survival_table = numpy.zeros((2, num_classes, num_brackets))
print survival_table

for ticket_class in xrange(num_classes):
    for fare_bracket in xrange(num_brackets):
        mask_ticket = data[...,2].astype(numpy.float) == ticket_class + 1
        mask_fare = (data[...,9].astype(numpy.float) >= fare_bracket_size * fare_bracket)&(data[...,9].astype(numpy.float) < fare_bracket_size * (fare_bracket+1))
        
        survival_table[0, ticket_class, fare_bracket] = data[(data[...,4] == "female")&mask_ticket&mask_fare, 1].astype(numpy.float).mean()
        survival_table[1, ticket_class, fare_bracket] = data[(data[...,4] == "male")&mask_ticket&mask_fare, 1].astype(numpy.float).mean()

survival_table[survival_table != survival_table] = 0.
print survival_table

survival_table[survival_table >= 0.5] = 1.
survival_table[survival_table < 0.5] = 0.
print survival_table

with io.open("data/test.csv", "rb") as test_in:
    csv_in = csv.reader(test_in)
    print csv_in.next()
    
    with io.open("data/test_gender_class.csv", "wb") as test_out:
        csv_out = csv.writer(test_out)
        csv_out.writerow(["PassengerId", "Survived"])
        
        for row in csv_in:
            #print row
            ticket_class = int(row[1]) - 1
            fare_bin = min(3, int(float(row[8] or 0) / fare_bracket_size))
            mf = 0
            if row[3] != "female":
                mf = 1
            
            # print mf, ticket_class, fare_bin
            csv_out.writerow([row[0], str(int(survival_table[mf, ticket_class, fare_bin]))])
