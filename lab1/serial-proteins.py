import time
import pandas as pd
import matplotlib.pyplot as plt
import sys


def read_pattern():
    pattern = input('Input the pattern: ')
    return pattern

def uppercase(x):
    return x.upper().strip()

def number_ocurrences(sub, s):
    # string.count() does not overlap the pattern
    return s.count(sub)

def review_protein_row(p, row):
    ocur_item = None
    id = row['structureId']
    seq = row['sequence']
    if seq.count(p) > 0:
        number = seq.count(p)
        ocur_item = [id, number]
    return ocur_item


if __name__ == '__main__':
    # 1.- Read the pattern to search from the keyboard. 
    pattern = read_pattern()
    if not pattern:
        print('Pattern empty.')
        sys.exit()
    
    # 2.- Changes the pattern to UPPERCASE
    pattern = uppercase(pattern)
    print(f'My pattern is: {pattern}')
    
    # 3.- Start execution time
    t_start = time.time()
    
    # 4.- Reads the protein patterns from the file in pairs (id, sequence)
    ocurrences = []
    CHUNKSIZE = 100000
    reader = pd.read_csv('proteins.csv', dtype={'structureId':int, 'sequence':str}, chunksize=CHUNKSIZE)
    t_now = time.time()
    read_time = t_now - t_start
    print(f'Read time: {read_time*1000} ms')
    
    # 5. Look for occurrences of the pattern string inside each protein sequence
    # 6.- If there are occurrences, register the id of the protein and the number of occurrences in the sequence
    for df in reader:
        df['ocurrences'] = df['sequence'].map(lambda sequence : number_ocurrences(pattern, sequence))
        df = df[df.ocurrences > 0]
        ocurrences.append(df)
        
    # 7.- Show the execution time
    t_now = time.time()
    exec_time = t_now - t_start
    print(f'Execution time: {exec_time*1000} ms')
    
    # 8.- Print a histogram of occurrences using protein id as X and number of ocurrences as Y, using matplotlib.pyplot. Represent the 10 proteins with more matches.
    if ocurrences:
        df_total = pd.concat(ocurrences)
        df_max = df_total.sort_values(by=['ocurrences'], ascending=False).head(10).sort_values(by=['structureId']).set_index('structureId')
        df_max.plot(kind='bar', figsize=(10,5))
        plt.show()
        i_max = df_max['ocurrences'].idxmax()
        ocur_max = df_max['ocurrences'][i_max]
        print(f'Protein with most matches: - ID: {i_max}, OCURRENCES: {ocur_max}')
    else:
        print('No ocurrences.')
