import sys
import pickle
#detectron venv activate

sys.path.append("detectron2/projects/DensePose/")

set = "train"
target_file = "./LIP/anotations/dense_anotations/train_combined.pkl"
n = 11

d = dict()
for i in range (1, n+1):
    print('Loading {}_{} ....'.format(set, i))
    f = open('./LIP/anotations/dense_anotations/dense_{}_{}.pkl'.format(set, i), 'rb')
    data = pickle.load(f)

    print('Processing {}_{} ....'.format(set, i))
    for tmp in data:
        if(len(tmp['scores']) > 0):
            tmp['scores'] = tmp['scores'][0]
            tmp['pred_densepose_uv'] = tmp['pred_densepose'][0].uv                    #UV
            tmp['pred_densepose_labels'] = tmp['pred_densepose'][0].labels  #I
            tmp['pred_boxes_XYXY'] =  tmp['pred_boxes_XYXY'][0]
            tmp['pred_densepose'] = None

            d[tmp['file_name']] = tmp

    print('Deleting {}_{} ....'.format(set, i))
    del data

print('Saving pickles {}_{} length:{}....'.format(set, i, len(d)))
with open(target_file, 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

