import io
import csv

with io.open("data/test.csv", "rb") as test_in:
    csv_in = csv.reader(test_in)
    headers = csv_in.next()
    
    with io.open("data/test_gender.csv", "wb") as test_out:
        csv_out = csv.writer(test_out)
        csv_out.writerow(["PassengerId", "Survived"])
        
        for row in csv_in:
            if row[3] == "female":
                csv_out.writerow([row[0], "1"])
            else:
                csv_out.writerow([row[0], "0"])

