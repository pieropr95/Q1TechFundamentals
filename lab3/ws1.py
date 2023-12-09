import requests
import json

from datetime import datetime

import matplotlib.pyplot as plt

def get_balance_data(months=12, decimals=1, version='2.1', lang='en'):
    
    if lang not in ['en', 'fr', 'de']:
        print('Language not available')
        lang='en'

    # CREATE HTTP REQUEST
    base_url = f'http://ec.europa.eu/eurostat/wdds/rest/data/v{version}/json/{lang}/ei_bsco_m'
    filters = f'?indic=BS-CSMCI&precision={decimals}&lastTimePeriod={months}&geo=EU27_2020&unit=BAL&s_adj=SA'
    url = base_url + filters
    print(url)

    # SEND GET REQUEST AND WAIT FOR A RESPONSE
    r = requests.get(url)
    print(r.status_code)

    # ONLY IF THE REQUEST IS SUCCESSFUL - RETURN JSON
    if r.status_code not in range(200, 299):
        print(r.text)
        return {}
    return json.loads(r.text)


def main():
    
    # GET BALANCE DATA IN THE FORM OF JSON VARIABLE
    json_balance = get_balance_data()
    if not json_balance:
        print('JSON response could not be obtained.')
        return
    
    # GET BALANCE VALUES OF THE LAST 12 MONTHS
    values = list(json_balance['value'].values())
    # values.reverse()
    print(values)

    # GET THE LAST 12 MONTHS
    time = list(json_balance['dimension']['time']['category']['label'].keys())
    # time.reverse()
    date = [datetime.strptime(t, '%YM%m').strftime('%Y-%m') for t in time]
    print(time)
    print(date)

    # PLOT THE DATA
    # plt.bar(date, height=values, color='r', width=0.4)
    plt.plot(date, values, color='r')
    plt.fill_between(date, values, color='r', alpha=0.4)
    plt.xticks(rotation=45)
    plt.title('Balance data for Consumer Confidence Indicator')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
