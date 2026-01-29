import json
import sys

fin = open(sys.argv[1], 'r')
j = json.load(fin)
fin.close()
#j['episodes'] = j['episodes'][:1]
if j['episodes'][0]['scene_id'].startswith('data'):
    j['episodes'][0]['scene_id'] = '/home/kirill/habitat-lab/' + j['episodes'][0]['scene_id']
fout = open(sys.argv[1], 'w')
json.dump(j, fout)
