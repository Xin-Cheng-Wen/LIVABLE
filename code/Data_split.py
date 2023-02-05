import pandas as pd
import json
from tqdm import tqdm
from pandas.core.frame import DataFrame

fp = open('dataset/devign_input/devign-line-ggnn.json')
train_data = json.load(fp)
print(len(train_data))

data = DataFrame(train_data)

vul = []
non_vul = []s
for i in tqdm(train_data):
    dic = {}
    dic['node_features'] = i['node_features']
    dic['graph'] = i['graph']
    dic['targets'] = i['targets']
    if dic['targets'][0][0] == 1:
        vul.append(dic)
    elif dic['targets'][0][0] == 0:
        non_vul.append(dic)
print(len(vul))
print(len(non_vul))


vul = DataFrame(vul)
non_vul = DataFrame(non_vul)

train = vul[:7358].append(non_vul[:8973])
valid = vul[7358:8458].append(non_vul[8973:10094])
test = vul[8458:].append(non_vul[10094:])


train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)
valid = valid.sample(frac=1).reset_index(drop=True)

train_file = open('Devign/devign_input/devign_train_GGNNinput.json', 'w')
test_file = open('Devign/devign_input/devign_test_GGNNinput.json', 'w')
valid_file = open('Devign/devign_input/devign_valid_GGNNinput.json', 'w')

train.to_json('Devign/devign_input/devign_train_GGNNinput.json', orient='records',lines=True)
test.to_json('Devign/devign_input/devign_test_GGNNinput.json', orient='records', lines=True)
valid.to_json('Devign/devign_input/devign_valid_GGNNinput.json', orient='records', lines=True)