# Rasched - did not want to change main.py but just followed same structure
import torch
import pandas as pd 
from tsai.all import InceptionTime
import numpy as np
from file_io import preprocess_data
import pandas as pd
import os

class Inference:

    def __init__(self, filepath, model_path):

        # device
        # if torch.cuda.is_available():
        #     print("Using GPU")
        #     self.device = torch.device('cuda:0')
        # else:
        #     print("Using CPU")
        self.device = torch.device('cpu')

        self.filepath = filepath
        self.model_path = model_path

        self.model = InceptionTime(1,2).to(self.device)
        self.model_state_dict = torch.load(self.model_path)['model_state_dict']
        self.model.load_state_dict(self.model_state_dict)
        
        print('Model successfully loaded!')

    def process_input(self, array):
        
        array = preprocess_data(array)
        array = torch.from_numpy(array.copy()).float()
        array = torch.unsqueeze(array,0)
        return array.to(self.device)

    # Rasched - loads the excel file that Jagadish gave me, I included the trace limit since the training cut it off at 1200
    # I kept xls loader and process_input seperate in case the file loading changes so one function only needs to be added
    def xls_loader(self, X = None, trace_limit = 1500):

        if X is None:
            X = torch.tensor([]).to(self.device)

        df = pd.read_excel(self.filepath, header=None).iloc[:trace_limit]
        original_array = df.to_numpy().T

        # tranposed so that every index of the numpy array is a trace
        for item in original_array:
            processed_item = self.process_input(item)
            X = torch.cat((X, processed_item.unsqueeze(0)), 0)

        return X, original_array

    def inference(self, correct_label=None):

        pred_labels = []
        pred_confidences = []
        final_labels = [] # every 4 tracks are one set, we take the most likely one to be complimentary
        # complimentary_tracks = []

        inference_dataset , _ = self.xls_loader()
        
        with torch.no_grad():
            pred_label = self.model(inference_dataset)
            pred_confidences.extend(torch.nn.functional.softmax(pred_label, dim=1).tolist())
            pred_labels.extend(pred_label.data.cpu().argmax(dim=1).numpy().tolist())


        final_data = []

        for i in range(0,len(pred_confidences), 4):
            
            probabilities = np.array(pred_confidences[i:i+4])[:,1]
            # we need the one thats most likely to be complimetary so greatest value in the 1th element
            row_data = {
                'label from 0 to 3': np.argmax(probabilities),
                'prob0': probabilities[0],
                'prob1': probabilities[1],
                'prob2': probabilities[2],
                'prob3': probabilities[3]
            }

            final_data.append(row_data)
            final_labels.append(np.argmax(probabilities))

            # these labels correspond to the column number in the excel in batches of 4
            # complimentary_tracks.append()

        
        pd.DataFrame(final_data).to_csv(self.filepath[:-5] +'.csv', index=False)
        
        pred_confidences = np.array(pred_confidences).max(axis=-1).tolist()
        average_confidence = round(np.mean(pred_confidences),3)*100
        print('Average confidence', average_confidence)

        if correct_label is not None:
            accuracy = np.sum(np.array(final_labels)==correct_label)/len(final_labels)
            print('Accuracy:', round(accuracy,3)*100)
        


if __name__ == '__main__':

    directory_path = 'Z:/Rasched/GAPSEQML/data/IPE'
    MODEL_PATH = 'Z:/Rasched/GAPSEQML/models/ss_w_0.15_of_each_final_for_now/inceptiontime_model_230916_1330'

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.xlsx') and not file.startswith('~'):

                path = os.path.join(root, file)
                instance_ = Inference(path, MODEL_PATH)
                # 0,1,2,3 are the columns in the excel file corresponding to each trace - see Jagadish for correct label
                # MAKE SURE THE CORRECT LABEL IS IN THAT ORDER  
                instance_.inference()





















