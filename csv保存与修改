import csv
with open(doc_answer,'rb') as fp_in, open('newfile.csv','wb') as fp_out:
    reader = csv.reader(fp_in)
    writer = csv.writer(fp_out)
    header = 0
    for row in reader:        
        if row[-1]==row[-4] or header == 0:
            writer.writerow(row)
        header = 1  
