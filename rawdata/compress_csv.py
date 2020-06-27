import argparse
import csv
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, help='working directory')
# parser.add_argument('-f', '--channel_from', type=int, help='desired min channel')
# parser.add_argument('-t', '--channel_to', type=int, help='desired max channel (not contain')
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
            newrow = []
            newrow.append(row[0])  ## time stamp
            newrow.append(row[1])  ## channel 0
            newrow.append(row[3])  ## channel 2
            # newrow = [row[0],] + row[args.channel_from:args.channel_to]
            newfile.append(newrow)

    with open(os.path.join(args.dir, filename), 'w') as file:
        data = csv.writer(file)
        data.writerow(['AD7770',])
        for row in newfile:
            data.writerow(row)
    

