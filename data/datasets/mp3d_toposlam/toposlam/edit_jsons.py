import os
import json

for fn in os.listdir('.'):
    if fn.endswith('.json'):
        fin = open(fn, 'r')
        json_text = fin.read()
        fin.close()
        json_text = json_text.replace('/home/kirill/habitat-lab/data', '/data')
        json_text = json_text.replace('"data', '"/data')
        fout = open(fn, 'w')
        fout.write(json_text)
        fout.close()