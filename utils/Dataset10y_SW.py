import torch
from torch.utils.data import Dataset
import pandas as pd
from glob import glob
import numpy as np



def load_parquet(seq):
    data = pd.read_parquet(seq)

    data.drop(columns=['id_grid', 'index',  'randomID','AN_ORIGINE', 'ORIGINE','id_left', 'ared', 'use', 'RandomV','index_righ', 'NBR_1', 'NBR_2','NBR_3','NBR_4','NBR_5','NBR_6','NBR_7', 'NBR_8', 'NBR_9', 'NBR_10','iNBR50','iNBR100','iNBR125','iNBR150','iNBR200','geometry', 'randomAN', 'slope', 'theilslope'], inplace=True)
    data['Xseq'] = data['Xseq']+1
    data['Xseq'] = data['Xseq'].replace(100, 0)
    data=data.replace(-32768.0000, np.nan)
    return data


def numpy_fill(a, startfillval=0):
    """
    Fill sequence with the previous year value when the pixel are null.
    https://stackoverflow.com/questions/62038693/numpy-fill-nan-with-values-from-previous-row
    Args:
        a : array corresponding to the sequence
        startfillval : 0

    Returns:
       Array without nan value
    """
    mask = np.isnan(a)
    tmp = a[0].copy()
    a[0][mask[0]] = startfillval
    mask[0] = False
    idx = np.where(~mask,np.arange(mask.shape[0])[:,None],0)
    out = np.take_along_axis(a,np.maximum.accumulate(idx,axis=0),axis=0)
    fill_values=np.where(np.all(out[:2] == 0, axis=0), out[2], out[:2])
    out[:2]= fill_values
    out[0]= np.where(np.all(out[:1] == 0, axis=0), out[1], out[0])
    a[0] = tmp
    return out

#Training dataset
class MultiDisDataset(Dataset):

         def __init__(self, seqfile) :
             self.seqfile = load_parquet(seqfile)

             self.disId_data =  self.seqfile.iloc[:,2]
             self.disId = np.asarray(self.disId_data.values).astype(np.float)

             self.disX_data =  self.seqfile.iloc[:,4]
             self.disX = np.asarray(self.disX_data.values).astype(np.float)

             self.disY_data =  self.seqfile.iloc[:,5]
             self.disY = np.asarray(self.disY_data.values).astype(np.float)

             self.X_data = self.seqfile.iloc[:,6:106]#31=2+6band*10y
             self.X = self.X_data.values
             self.X = np.asarray(self.X, dtype='float32')
             self.seq = self.X.reshape(self.X.shape[0],int(self.X.shape[1]/6),6)


             self.Y_type = self.seqfile.iloc[:,0]  #For type
             self.Y_dur = self.seqfile.iloc[:,3]  #For severity
             self.Y_date = self.seqfile.iloc[:, 1]  #For date


             self.Y_type= np.asarray(self.Y_type.values, dtype='float32')
             self.Y_dur= np.asarray(self.Y_dur.values, dtype='float32')
             self.Y_date= np.asarray(self.Y_date.values, dtype='float32')


         def __len__(self):
             return len(self.seq)
         def __getitem__(self, idx):
             labels_type=self.Y_type[idx]
             labels_date=self.Y_date[idx]
             labels_dur=self.Y_dur[idx]
             sequence = self.seq[idx]
             #print('sequence', sequence)
             sequence[sequence == (-9999)] = np.nan
             sequence= numpy_fill(sequence)
             #print('sequence_', sequence)

             data = {'sequence':torch.Tensor(sequence)/10000,
                    'labels': {'label_type':torch.tensor(labels_type).long(),
                               'label_dur':torch.tensor(labels_dur).long(),
                               'label_date':torch.tensor(labels_date).long()}}

             id = self.disId[idx]
             x = self.disX[idx]
             y = self.disY[idx]
             return data, id, x, y


# Dataset Class for inference (without label)
# Input are dataframe created from slidding windows images with x,y coordinate and spectral temporal value
class Dataset_inf(Dataset):
         def __init__(self, seqdf) :
             self.seqfile = seqdf
             #print(seqdf.shape)

             self.year =  self.seqfile.iloc[:,60] #Get x coordinate
             self.year = np.asarray(self.year).astype(float)

             self.coordx =  self.seqfile.iloc[:,61] #Get x coordinate
             self.coordx = np.asarray(self.coordx).astype(float)

             self.coordy =  self.seqfile.iloc[:,62] #Get y coordinate
             self.coordy = np.asarray(self.coordy).astype(float)

             self.nancpt =  self.seqfile.iloc[:,63] #Get nancount
             self.nancpt = np.asarray(self.nancpt).astype(float)

             self.X_data = self.seqfile.iloc[:,:60]/10000  #31=1+6band*7y
             self.X = self.X_data.values
             self.X = np.asarray(self.X, dtype='float32')

             self.seq = self.X.reshape(self.X.shape[0],int(self.X.shape[1]/6),6) #reshape sequence as numpy of size (37,6)




         def __len__(self):
             return len(self.seq)
         def __getitem__(self, idx):

             sequence = self.seq[idx]
             #print('brut',sequence)

             sequence[sequence ==-2000/10000] = np.nan
             sequence[sequence ==-32768/10000] = np.nan
             sequence[sequence == 65535/10000] = np.nan
             sequence[sequence == 16022.125 / 10000] = np.nan

             sequence= numpy_fill(sequence) #fill nan value
             # Organize the dataset into dictionnary
             data = {'sequence':torch.Tensor(sequence),
                     }

             cX = self.coordx[idx]
             cY = self.coordy[idx]
             nan_= self.nancpt[idx]
             cYear= self.year[idx]


             return data, nan_, cX, cY, cYear


if __name__=="__main__":
    # Test for visualization
    seqfile = '/Dataset/k/10.0.parquet'
    validfile = glob(seqfile,  recursive=True)
    tt = Dataset(validfile)
    fig = plt.figure(figsize=(15,4))
    i=0
    for seq in tt:
        i +=1
        data, id = seq
        ax = fig.add_subplot(2,4,i)
        ax.plot((data['sequence']))
