
import numpy as np
import os
from ska_sdc import Sdc2Scorer

#This script can be used to score any of the catalog from the archive

if(not os.path.isfile("sky_ldev_truthcat_v2.txt")):
	os.system("wget --content-disposition https://www.dropbox.com/scl/fo/e847a6pnjtqk7xmxz6flt/AIo4631BbKfwhW2NGhvtbkw/sky_ldev_truthcat_v2.txt?rlkey=84kkeaw021ajh7t9lqur2n7p8")
if(not os.path.isfile("sky_full_truthcat_v2.txt")):
	os.system("wget --content-disposition https://www.dropbox.com/scl/fo/ce4o2tqhy2ddkowwecs5z/ADaaQzoKGwkHRkdPLZURw1M/sky_full_truthcat_v2.txt?rlkey=ywcljxh6eq2q18r629prn5w0j")


sub_cat_path = "../../catalogs/YOLO_CIANNA_catalog_SDC2_BT1_sc25453_MINERVA_Cornu2026.txt"
truth_cat_path = "sky_full_truthcat_v2.txt"

scorer = Sdc2Scorer.from_txt(sub_cat_path, truth_cat_path, sub_skiprows=0, truth_skiprows=0)
scorer.run(detail=True)

print("Final score: {}".format(scorer.score.value))
print ("Ndet:", scorer.score.n_det, "\nNmatch:", scorer.score.n_match, "\nNfalse:", scorer.score.n_bad + scorer.score.n_false)
print ("Avg characterization score:",scorer.score.acc_pc, "\nPurity:", (scorer.score.n_match/(scorer.score.n_match+scorer.score.n_false)))

score_details = scorer.score.scores_df

print ("Avg subscores:")
print ("Position:",np.mean(score_details["position"]))
print ("Central frequency:",np.mean(score_details["central_freq"]))
print ("Line flux", np.mean(score_details["flux"]))
print ("HI size",np.mean(score_details["hi_size"]))
print ("Line width",np.mean(score_details["w20"]))
print ("PA",np.mean(score_details["pa"]))
print ("Inclination",np.mean(score_details["i"]))
