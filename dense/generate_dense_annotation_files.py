import sys
import pickle
import json
import numpy
#detectron venv activate

sys.path.append("detectron2/projects/DensePose/")

dataset = "valid"
target_dir = "./LIP/anotations/dense_anotations/valid"
n = 4

s = set()
for i in range (1, n+1):
    print('Loading {}_{} ....'.format(dataset, i))
    f = open('./LIP/anotations/dense_anotations/dense_{}_{}.pkl'.format(dataset, i), 'rb')
    data = pickle.load(f)

    print('Processing {}_{} ....'.format(dataset, i))
    for tmp in data:
        if(len(tmp['scores']) > 0):
            tmp['scores'] = tmp['scores'][0].cpu().numpy().tolist()
            tmp['pred_densepose_uv'] = tmp['pred_densepose'][0].uv.cpu().numpy().tolist()                    #UV
            tmp['pred_densepose_labels'] = tmp['pred_densepose'][0].labels.cpu().numpy().tolist()  #I
            tmp['pred_boxes_XYXY'] =  tmp['pred_boxes_XYXY'][0].cpu().numpy().tolist()
            tmp['pred_densepose'] = None
            tmp['file_name'] = tmp['file_name'].split('/')[-1]

            name = tmp['file_name'].split('/')[-1]
            s.add(name)

            jsonStr = json.dumps(tmp)
            with open('{}/{}.txt'.format(target_dir, name), 'w') as outfile:
                json.dump(jsonStr, outfile)

    print('Deleting {}_{} ....'.format(dataset, i))
    del data

# USE:
# with open('364458_211998.jpg.txt') as json_file:
#     data = json.load(json_file)
# tmp = json.loads(data)
# tmp['pred_boxes_XYXY']

