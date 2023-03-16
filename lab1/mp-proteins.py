import time
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp

def number_ocurrences(substring, string):
    return string.count(substring)

def task(pattern, df):
    # 5. Look for occurrences of the pattern string inside each protein sequence
    # 6.- If there are occurrences, register the id of the protein and the number of occurrences in the sequence
    df['sequence'] = [number_ocurrences(pattern, sequence) for sequence in df['sequence']]
    return df

def collect_result (partialresult):
    global results
    results.append(partialresult)

def taskprocess(pattern, df, results):
    # 5. Look for occurrences of the pattern string inside each protein sequence
    # 6.- If there are occurrences, register the id of the protein and the number of occurrences in the sequence
    df['sequence'] = [number_ocurrences(pattern, sequence) for sequence in df['sequence']]
    results.append(df)

if __name__ == "__main__":
    # 1.- Read the pattern to search from the keyboard. 
    pattern = input("Please, enter the pattern to search:\n")
    
    # 2.- Changes the pattern to UPPERCASE
    pattern = pattern.upper()
    
    # 3.- Start execution time
    start = time.time()
    
    # 4.- Reads the protein patterns from the file in pairs (id, sequence)
    numprocesses = mp.cpu_count() # CHANGEABLE PARAMETER
    chunksize = 100000 # CHANGEABLE PARAMETER
    
    # Read in parallel
    reader = pd.read_csv('proteins.csv', dtype={'structureId':int, 'sequence':str}, chunksize=chunksize)
    
    # OPTION 1: apply_async (faster)
    global results
    results = []
    p = mp.Pool(processes = numprocesses)
    for df in reader: 
        p.apply_async(task, args=(pattern, df), callback=collect_result) 
    p.close()
    p.join()
    df_total = pd.concat(results)
    
    # OPTION 2: starmap_async
    # global results
    # results = []
    # p = mp.Pool(processes = numprocesses)
    # p.starmap_async(task, [(pattern, df) for df in reader], callback=collect_result)
    # p.close()
    # p.join()
    # df_total = pd.concat(results[0])
    
    # OPTION 3: apply
    # p = mp.Pool(processes = numprocesses)
    # results = [p.apply (task, args=(pattern, df)) for df in reader]
    # df_total = pd.concat(results)
    # p.close()
    
    # OPTION 4: starmap
    # results = p.starmap(task, [(pattern, df) for df in reader])
    # p.close()
    # df_total = pd.concat(results)
    
    # OPTION 5: Process
    # manager = mp.Manager()
    # results = manager.list()
    # processes = []
    # for df in reader:
    #     process = mp.Process(target=taskprocess, args=(pattern, df, results,))
    #     processes.append(process)     
    # for process in processes:
    #     process.start()
    # for process in processes:
    #     process.join()
    # df_total = pd.concat(results)
    
    # COMMON TO ALL OPTIONS:
        
    # 7.- Show the execution time
    print("Execution time: " + str(time.time() - start))
    
    # 8.- Print a histogram of occurrences using protein id as X and number of ocurrences as Y, using matplotlib.pyplot. Represent the 10 proteins with more matches.
    df_max = df_total.sort_values(by=['sequence'], ascending=False).head(10).set_index('structureId')
    df_max.plot(kind='bar', figsize=(10,5))
    plt.ylabel('frequence')
    df_max.rename(columns={'sequence':'frequence'} ,inplace=True) 
    plt.show()
    print(df_max)
    
