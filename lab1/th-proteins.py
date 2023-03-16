import threading
import pandas as pd
import time
import matplotlib.pyplot as plt

def number_ocurrences(substring, string):
    return string.count(substring)

def task(pattern, df, lock):
    global results
    # 5. Look for occurrences of the pattern string inside each protein sequence
    # 6.- If there are occurrences, register the id of the protein and the number of occurrences in the sequence
    df['sequence'] = [number_ocurrences(pattern, sequence) for sequence in df['sequence']]
    lock.acquire()
    results.append(df)
    lock.release()
            
if __name__ == "__main__":
    # Implementation with Multi-threading
    global results
    results= []
    
    # 1.- Read the pattern to search from the keyboard. 
    pattern = input("Please, enter the pattern to search:\n")
    
    # 2.- Changes the pattern to UPPERCASE
    pattern = pattern.upper()
    
    # 3.- Start execution time
    start = time.time()
    chunksize = 50000 # CHANGEABLE PARAMETER
    
    # 4.- Reads the protein patterns from the file in pairs (id, sequence)
    reader = pd.read_csv('proteins.csv', dtype={'structureId':int, 'sequence':str}, chunksize=chunksize)
    th = []
    lock = threading.Lock()
    for i, chunk in enumerate(reader):
        th.append(threading.Thread(name='th%s' %i, target= task, args=(pattern, chunk, lock)))
        th[i].start()
        # th = threading.Thread(target= task, args=(pattern, chunk))
        # th.start()
    for i, chunk in enumerate(reader):
        th[i].join()
        # th.join()
    df_total = pd.concat(results) 
    
    # 7.- Show the execution time
    print("Execution time: " + str(time.time() - start))
    
    # 8.- Print a histogram of occurrences using protein id as X and number of ocurrences as Y, using matplotlib.pyplot. Represent the 10 proteins with more matches.
    df_max = df_total.sort_values(by=['sequence'], ascending=False).head(10).set_index('structureId')
    df_max.plot(kind='bar', figsize=(10,5))
    df_max.rename(columns={'sequence':'frequence'} ,inplace=True) 
    plt.ylabel('frequence')
    plt.show()
    # 9.- Print the id and number of the protein with max occurrences.
    print('ID and frequence of the protein with max ocurrences:')
    print(df_max.head(1))
    