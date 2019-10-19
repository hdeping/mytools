#!/usr/local/bin/python3

'''
Tool to change the .ris citation file format (used e.g. by Nature) to the bibtex format. Give the .ris file as input and optionally the .bib file to be used as output. 


author: K. Saaskilahti, 26.3.2015
'''

import sys


risfile = sys.argv[1]

author_list=[]
title=None
journal=None
volume=None
year=None
month=None
startingpage=None
finalpage=None
publisher=None
doi=None
abstract=None
url=None

with open(risfile,'r') as f:
    for data in f:
        data=data.split('-',1) 
        #print("data:",data)
        if len(data)==1:
            pass
        else:
            field=data[0].strip(' ')
            value=data[1].strip(' ').strip('\n').strip('\r')
            #print case
            if field == 'AU':
                author_list.append(value)
            elif field == 'TI':
                title=value
            elif field == 'JA' or field == 'JO':
                journal=value
            elif field == 'VL':
                volume=value
            elif field == 'PY':
                year=value.rsplit('/')[0]
                #month=value.rsplit('/')[1]
            elif field == 'SP':
                startingpage=value
            elif field == 'EP':
                finalpage=value
            elif field == 'L3' or field == 'DO':
                doi=value
            elif field == 'PB':
                publisher=value
            elif field == 'AB':
                abstract=value
            elif field == 'UR':
                url=value

# dealing with the data
lines=[]
firstauthor=author_list[0].rsplit(',')[0].strip(' ')

lines.append('@article{'+firstauthor.lower()+year+',')

authorline=' '*4 + 'author={' + ' and '.join(author_list)+'},'
lines.append(authorline)
if title is not None:
    lines.append(' '*4 + 'title={' + title + '},')
if journal is not None:
    lines.append(' '*4 + 'journal={' + journal + '},')
if volume is not None:
    lines.append(' '*4 + 'volume={' + volume + '},')
if startingpage is not None and finalpage is not None:
    lines.append(' '*4 + 'pages={' + startingpage + '--'+finalpage+'},')
if year is not None:
    lines.append(' '*4 + 'year={' + year + '},')

if doi is not None:
    lines.append(' '*4 + 'doi={' + doi + '},')
# publisher
if publisher is not None:
    lines.append(' '*4 + 'publisher={' + publisher + '},')
# abstract
if abstract is not None:
    lines.append(' '*4 + 'abstract={' + abstract + '},')
# url
if url is not None:
    lines.append(' '*4 + 'url={' + url + '}')
lines.append('}\n')

bibfile = risfile[:-4] + ".bib"
print('Writing output to file ',bibfile)
with open(bibfile,'w') as f:
    f.write('\n'.join(lines))
