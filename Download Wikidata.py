
# coding: utf-8

# In[1]:


import requests
import json
import regex as re
from glob import glob
import json


# In[2]:


query="""
SELECT ?itemLabel 
WHERE {
  ?item wdt:P31 wd:Q5 . #instance of human
        ?item wdt:P106/wdt:P279 wd:Q639669 . #occupation a subclass of musician
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
   }
}

LIMIT 10
"""

QID_REGEX=re.compile(r'^Q[0-9]+$')


url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
def get_query(queryfile):
    query = ""
    with open(queryfile) as fp:
        query = fp.read().strip()
    return query

def download_data(query, data_format="json"):
    if data_format == "xml":
        data = requests.get(url, params={'query': query, 'format': 'xml'})
    else:
        data = requests.get(url, params={'query': query, 'format': 'json'}).json()
        print "Downloaded %s records" % (len(data["results"]["bindings"]))
    return data

def save_data(data, outfile, label="itemLabel", data_format="json"):
    with open(outfile, "wb+") as fp:
        processed, written = 0, 0
        if data_format == "xml":
            from bs4 import BeautifulSoup
            data_xml = BeautifulSoup(data.text, "lxml")
            results = data_xml.find_all("result")
        else:
            results = data["results"]["bindings"]
            
        for k in results:
            if data_format == "xml":
                line = k.text.strip()
            else:
                line=k[label]["value"]
            try:                
                processed += 1
                if QID_REGEX.match(line):
                    continue
                print >> fp, line
                written += 1
            except:
                continue
    print "Processed: %s, Written: %s" % (processed, written)
    return True


# In[3]:


data = download_data(query=query)


# In[4]:


save_data(data, "data/wikidata/downloaded/temp.txt", label="itemLabel")


# In[5]:


query_files = glob("data/wikidata/queries/*.txt")
print query_files


# In[5]:


OUTDIR="./data/wikidata/downloaded/"


# In[6]:


for q in ["data/wikidata/queries/companynames.txt"]:
    print "Processing %s" % q
    if "persons" in q or "movies" in q:
        data_format="xml"
    else:
        data_format="json"
    base_file=q.split("/")[-1]
    outfile="%s/%s.results.txt" % (OUTDIR, base_file)
    query = get_query(q)
    data = download_data(query=query, data_format=data_format)
    print "Saving to %s" % outfile
    save_data(data, outfile, label="itemLabel", data_format=data_format)


# In[7]:


for q in ["data/wikidata/queries/buildings.txt"]:
    print "Processing %s" % q
    if "persons" in q or "movies" in q:
        data_format="xml"
    else:
        data_format="json"
    base_file=q.split("/")[-1]
    outfile="%s/%s.results.txt" % (OUTDIR, base_file)
    query = get_query(q)
    data = download_data(query=query, data_format=data_format)
    print "Saving to %s" % outfile
    save_data(data, outfile, label="itemLabel", data_format=data_format)


# In[8]:


for q in query_files:
    print "Processing %s" % q
    if "persons" in q or "movies" in q:
        data_format="xml"
    else:
        data_format="json"
    base_file=q.split("/")[-1]
    outfile="%s/%s.results.txt" % (OUTDIR, base_file)
    query = get_query(q)
    data = download_data(query=query, data_format=data_format)
    print "Saving to %s" % outfile
    save_data(data, outfile, label="itemLabel", data_format=data_format)


# In[9]:


query


# In[11]:


data = requests.get(url, params={'query': query, 'format': 'xml'})


# In[26]:


save_data(data, outfile=outfile, data_format="xml")

