import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Converter:
    def __init__(self,the_dataFrame):
        self.df = the_dataFrame 
        self.df.dropna(inplace= True)
        self.k = []

    def Convert_object_to_number(self):        
        for col in self.df.select_dtypes(include='object'):
            l = self.df[col].unique()
            l.sort()
            d = {k:v for v,k in enumerate(l)}    
            self.df[col] = self.df[col].map(d)
            self.k.append(d)
        
        # print(self.k)
        return  self.df

    def Convert_Req(self,d:dict):
        return [self.k[key][val] for key,val in d.items()]

    def Reconvert(self,rsp,index:int):
        for k,v in self.k[index].items():
            if v == rsp:
                return k
