import argparse
import csv
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, help='working directory')
parser.add_argument('-c', '--channel', type=int, help='desired channel CHANNEL')
args = parser.parse_args()

# print(args.dir)
# print(args.channel)

# filename = '20002.csv'

filenames = os.listdir(args.dir)
for filename in filenames:
    if not filename.endswith('.csv'):
        continue

    print('working on', filename)
    newfile = []

    with open(os.path.join(args.dir, filename)) as file:
        data = csv.reader(file)
        for row in data:
            newrow = row[args.channel]
            newfile.append([newrow,])

    with open(os.path.join(args.dir, filename), 'w') as file:
        data = csv.writer(file)
        data.writerow(['AD7770',])
        for row in newfile:
            data.writerow(row)
    

