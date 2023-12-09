import requests
import json

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

def get_name_stats(name, lang='en', version='v1'):
    
    # CREATE HTTP REQUEST
    base_url = f'https://api.idescat.cat/onomastica/{version}/nadons/dades.json'
    filters = f'?id={name}&class=t&lang={lang}'
    url = base_url + filters
    print(url)

    # SEND GET REQUEST AND WAIT FOR A RESPONSE
    r = requests.get(url)
    print(r.status_code)

    # ONLY IF THE REQUEST IS SUCCESSFUL - RETURN JSON
    if r.status_code not in range(200, 299):
        return {}
    return json.loads(r.text)


def main():
    
    # GET STAT INFORMATION OF MARIA (CODE = 40683)
    code_maria = '40683'
    json_names = get_name_stats(code_maria)
    
    # EXTRACT RELEVANT INFORMATION FROM JSON FILE
    total_names = json_names['onomastica_nadons']['ff']['f']
    arr_years = []
    arr_values = []

    # SAVE IN LISTS THE 5 VARIABLES WE GET (LIMITING THE LAST 5 YEARS)
    for n_year in total_names:
        year = n_year['c']
        if int(year) >= (datetime.now().year-5):
            arr_years.append(year)
            rnk_total = int(n_year['rank']['total'])
            rnk_sex = int(n_year['rank']['sex'])
            fq = int(n_year['pos1']['v'])
            w_total = float(n_year['pos1']['w']['total'])
            w_sex = float(n_year['pos1']['w']['sex'])
            arr_values.append([rnk_total, rnk_sex, fq, w_total, w_sex])

    print(arr_years)
    print(np.array(arr_values))

    # PLOT THE DATA
    plt.subplot(2, 3, 1)
    plt.grid(zorder=0)
    plt.bar(arr_years, height=np.array(arr_values)[:,0], color='b', width=0.4, zorder=3)
    plt.title('Ranking for total population')
    plt.subplot(2, 3, 2)
    plt.grid(zorder=0)
    plt.bar(arr_years, height=np.array(arr_values)[:,1], color='g', width=0.4, zorder=3)
    plt.title('Ranking for women')
    plt.subplot(2, 3, 3)
    plt.grid(zorder=0)
    plt.bar(arr_years, height=np.array(arr_values)[:,2], color='r', width=0.4, zorder=3)
    plt.title('Frequency')
    plt.subplot(2, 3, 4)
    plt.grid(zorder=0)
    plt.bar(arr_years, height=np.array(arr_values)[:,3], color='y', width=0.4, zorder=3)
    plt.title('Weight per thousand of total population')
    plt.subplot(2, 3, 5)
    plt.grid(zorder=0)
    plt.bar(arr_years, height=np.array(arr_values)[:,4], color='m', width=0.4, zorder=3)
    plt.title('Weight per thousand of women')
    plt.show()


if __name__ == '__main__':
    main()
