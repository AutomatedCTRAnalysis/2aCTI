#******************************************************************************
        # This code is used to scrape the individual jsons stored in the
        # data folder  'enterprise-attack' and 'ics-attack'. It reads the 
        # relevant information before storing them in the "dataset full" file
#****************************************************************************** 


from ast import keyword
import os
import json
from collections import defaultdict # generates default key if missing 
import pandas as pd

dataset = defaultdict(lambda: defaultdict(list))
tech_dict = {}

with open('malware.json') as malware_file:
    malwares = json.load(malware_file)

malware_urls = set()

for malware in malwares:
    for url in malware['urls']:
        malware_urls.add(url)
print("malware url count", len(malware_urls))

# loop through enterprise and ics attack-pattern:
root_folder = '../data/'
folder_names = ['enterprise-attack-pattern', 'ics-attack-pattern']

for folder_name in folder_names: 

    folder = os.path.join(root_folder, folder_name)

    for filename in os.listdir(folder): 
        if filename.endswith('.json'):
            with open(os.path.join(folder, filename)) as file:
                file_json = json.load(file)["objects"][0] 
                # retrieve information:
                urls = []
                tactic_names = []

                tech_name = file_json['name'] # technique name
                tech_id = file_json['external_references'][0]['external_id'] # retrieve technique id
                mitre_domain = file_json['x_mitre_domains']
                description = file_json['description']
                
                for tactic in file_json['kill_chain_phases']: # tactic name 
                    tactic_names.append(tactic['phase_name']) 
            
                for ref in file_json['external_references'][1:]:
                    if 'url' in ref: # check if source has url 
                        url = ref['url'] # retrieve url 

                        # filtering out unecessary reports:
                        #filtering = ['microsoft', 'apple', 'github', 'wikipedia', \
                         #   'support.office', 'amazon', 'gitlab', 'capec', 'docker', 'youtube', 'google', 'mitre', 'zip']
                        
                        #if filtering not in url:  
                        urls.append(url)

                        dataset[url]['url'] = url
                        for domain in mitre_domain:
                            if domain not in dataset[url]['mitre_domain']:
                                dataset[url]['mitre_domain'].append(domain)
                        for tactic_name in tactic_names:
                            if tactic_name not in dataset[url]['tactic_name']:
                                dataset[url]['tactic_name'].append(tactic_name)
                        dataset[url]['tech_name'].append(tech_name)
                        dataset[url]['tech_id'].append(tech_id)
                        

                tech_dict[tech_id]= {
                'name':tech_name,
                'tactic': tactic_names,
                'filename' : filename,         
                'mitre_domain': mitre_domain,
                'urls':urls
            } 


name = 'dataset_full'
pd.DataFrame(dataset.values()).to_csv(name+'.csv')
with open(name+'.json', 'w') as file:
    json.dump(dataset, file, indent=4)
