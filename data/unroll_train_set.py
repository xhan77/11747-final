import csv

with open('train.txt', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    lines = []
    for line in reader:
        lines.append(line)

with open('unrolled_train.txt', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter='\t')
    for line in lines:
        csvwriter.writerow(line)
    for line in lines:
        csvwriter.writerow([line[1], line[0]])
