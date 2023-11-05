import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys

#Require installation of pycocotools from this repo https://github.com/cocodataset/cocoapi

load_epoch = 0
if (len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])

annType = "bbox"
prefix = "instances"

dataDir = "./"
dataType = "val2017"
annFile = "%s/annotations/%s_%s.json"%(dataDir,prefix,dataType)
cocoGt = COCO(annFile)

resFile="fwd_res/pred_%04d.json"%(load_epoch)
cocoDt=cocoGt.loadRes(resFile)

imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[:]

cocoEval = COCOeval(cocoGt,cocoDt,annType)
#cocoEval.params.catIds = [1] #To select class
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()



