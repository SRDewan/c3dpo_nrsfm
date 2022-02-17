import os
import numpy as np
import json
import random

def getJSON(fileObj):
	data = {}
	data['data'] = []

	for idx in range(len(fileObj['imgs'])):
		dataObj = {}
		keypts = np.load(fileObj['keypts'][idx])

		dataObj['img'] = fileObj['imgs'][idx]
		dataObj['kp_loc'] = keypts.T[:2, :].tolist()
		dataObj['kp_vis'] = keypts.T[2, :].tolist()
		dataObj['K'] = [
			[888.88,     0, 320],
			[     0,  1000, 240],
			[     0,     0,   1],
		]

		data['data'].append(dataObj)
	
	data['data'] = random.sample(data['data'], 10)
	data['dataset'] = "cars"
	return data 

def getFiles(rootDir):
	ret = { 'imgs': [], 'masks': [], 'keypts': []}

	for dirPath, dirs, files in os.walk(rootDir):
		for file in files:
			if "Color_00.png" in file and (file[:-12] + "Mask_00.png") in files and (file[:-12] + "KeyPoints.npy") in files:
				ret['imgs'].append(os.path.join(dirPath, file))
				ret['masks'].append(os.path.join(dirPath, file[:-12] + "Mask_00.png"))
				ret['keypts'].append(os.path.join(dirPath, file[:-12] + "KeyPoints.npy"))
           
	return ret

def main():
	fileObj = getFiles('../train_actual_small')
	data = getJSON(fileObj)
	print(json.dumps(data))

if __name__ == '__main__':
	main()
