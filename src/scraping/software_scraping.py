#******************************************************************************
        # This code is used to scrape the "software" page from the mitre website
        # to retrieve every software ID and its associated name.
#****************************************************************************** 



import requests,time, json
from bs4 import BeautifulSoup
from collections import defaultdict 

if __name__ == "__main__":
    attack_list = {}
    response = requests.get('https://attack.mitre.org/software/')
    if response.status_code == 200:
        Soup = BeautifulSoup(response.content,'html.parser')
        for attack in Soup.find_all('div', {'class': 'sidenav'})[1:]:
            name = attack.find('a').text.strip()   
            iD = attack.find('a').get('href').split('/software/')[1].replace('/', '')
            attack_list[iD] = name
        pass
        print(attack_list)  
    pass
    requests.Session.get()
    name = 'software_dataset'
    # Json for tactic info
    with open(name +'.json', 'w') as file:
        json.dump(attack_list, file, indent = 4)
pass