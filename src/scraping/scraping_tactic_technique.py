#******************************************************************************
        # This code is used to scrape the matrices for both tactic and techniques
        # from the mitre website to retrieve every tactic id and its associated 
        # name, description and techniques used. It stores tactic information in 
        # the "tactic dataset" json and the technique information in the "technique
        # dataset" json files
#****************************************************************************** 

import requests,time, json
from bs4 import BeautifulSoup
from collections import defaultdict 

# retrieve each technique from tactic page: 
def get_technique(link,types):
    techID_List = []
    tech_request = requests.get(link)
    if tech_request.status_code == 200:
        tech_Soup = BeautifulSoup(tech_request.content, 'html5lib')
        tech_Table = tech_Soup.find('table', {'class': 'table-techniques'})
        if types == 'tactics':
            for td in tech_Table.find_all('tr', {'class': 'technique'}):
                if td.find('td').text.strip().__contains__('T'):
                    techID_List.append((td.find('td').text.strip()))
                pass
            pass
            return techID_List
        else: 
            for td in tech_Table.find_all('tr', {'class': 'technique'}):
                if td.find_all('td')[0].text.strip().__contains__('T'):
                    iD = td.find_all('td')[0].text.strip()
                    dataset_technique[iD]['Technique_ID'] = iD
                    dataset_technique[iD]['Technique_Name'].append(td.find_all('td')[1].text.strip())
                    dataset_technique[iD]['Description'].append(td.find_all('td')[2].text.strip())
                pass
            pass
        pass
    pass

def retrieve_info(types):
    response = requests.get(urls[url])
    if response.status_code == 200:
        Soup = BeautifulSoup(response.content,'html5lib')
        MainTable = Soup.find('table',{'class':'table-alternate'}).find('tbody')
        for mt in MainTable.find_all("tr"):
            iD = mt.find_all('td')[0].text.strip()
            if types == 'tactics':
                dataset_tactic[iD]['Tactic_ID'] = iD
                
                dataset_tactic[iD]['Link'].append('https://attack.mitre.org/' + 
                mt.find_all('td')[0].find('a').get('href').strip())
                
                dataset_tactic[iD]['Tactic_Name'].append(mt.find_all('td')[1].text.strip())
                dataset_tactic[iD]['Description'].append(mt.find_all('td')[2].text.strip())
                
                dataset_tactic[iD]['Technique_ID'].append(get_technique('https://attack.mitre.org/' + 
                mt.find_all('td')[0].find('a').get('href').strip(), types))
        pass
pass

if __name__ == "__main__":
    dataset_tactic = defaultdict(lambda: defaultdict(list))
    dataset_technique = defaultdict(lambda: defaultdict(list))
    urls =['https://attack.mitre.org/tactics/enterprise/','https://attack.mitre.org/tactics/ics/', 
    'https://attack.mitre.org/techniques/enterprise/','https://attack.mitre.org/techniques/ics/']
    url = 0
    while url < len(urls):
        remain = len(urls) - url + 1
        print('\t\t\t loading:{} & remaining:{}'.format(url + 1, remain-1))
        if urls[url].__contains__('tactics'):
            retrieve_info('tactics')
        else:
            get_technique(urls[url],'techniques')
        url += 1
    pass

    # Store information: 
    name_tac = 'tactic_dataset'
    name_tec = 'technique_dataset'
    # Json for tactic info
    with open(name_tac +'.json', 'w') as file:
        json.dump(dataset_tactic, file, indent = 4)
    # Json for technique info: 
    with open(name_tec +'.json', 'w') as file:
        json.dump(dataset_technique, file, indent = 4)
pass

