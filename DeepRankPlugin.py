import numpy as np
import pandas as pd

def read_score_dict(data_dir, score_file_affix, unique_pids, print_labels=False, outfile=None):
    """
    Read the scores file performed by DeepRank
    """
    score_dict = {}
    statistics_dict={"Sample":[], "N_pos":[], "N_neg":[]}
    for pid in unique_pids:
        score_file = f"{data_dir}/{pid}/{pid}.{score_file_affix}"
        all_scores = []
        for line in open(score_file).readlines():
            curr_model, score = line.strip('\n').split('\t')
            model_id = curr_model.split('_')[0] + '-' + curr_model.split('_')[1]
            score = float(score)
            score_dict[model_id] = score
            all_scores.append(score)
        if print_labels:
            print(f"Sample {pid}: {int(np.sum(all_scores))} positives out of {len(all_scores)} models")
            statistics_dict['Sample'].append(pid)
            statistics_dict['N_pos'].append(int(np.sum(all_scores)))
            statistics_dict['N_neg'].append(len(all_scores) - int(np.sum(all_scores)))
    if print_labels:
        df=pd.DataFrame(statistics_dict)
        #df.to_csv("capri_dataset.csv")
        df.to_csv(outfile)
    return score_dict

import PyIO
import PyPluMA
import pickle
class DeepRankPlugin:
    def input(self, inputfile):
        self.parameters = PyIO.readParameters(inputfile)
    def run(self):
        pass
    def output(self, outputfile):
        CAPRI_DIR = PyPluMA.prefix()+"/"+self.parameters["inputdir"]
        toolsfile = PyPluMA.prefix()+"/"+self.parameters["toolsfile"]
        uniquepidfile = open(PyPluMA.prefix()+"/"+self.parameters["uniquepid"], "rb")
        unique_pids = pickle.load(uniquepidfile)
        tools = PyIO.readSequential(toolsfile)
        for tool in tools:
            mydict = read_score_dict(CAPRI_DIR, 'label', unique_pids, print_labels=True, outfile=outputfile+"."+tool+".labels.csv")
            outfile = open(outputfile+"."+tool+".dict.csv", 'w')
            for key in mydict.keys():
                outfile.write(key)
                outfile.write(',')
                outfile.write(str(mydict[key]))
                outfile.write('\n')
