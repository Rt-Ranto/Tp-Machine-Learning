import pandas as pd

import pydotplus
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from convertObjectToNumber import Converter

warnings.filterwarnings('ignore')

def prediction(df,le_req:str):

    # convertion des lettres non et oui en 1 et 0
    conv = Converter(df)
    df = conv.Convert_object_to_number()
    
    req = le_req.split(",")
    vreq = [x.strip() for x in req]
    features = [f'{k[:k.index(':')]}' for k in vreq]
    col = [colonne for colonne in df.columns]
    pred_req = [f'{k[k.index(':')+1:]}' for k in vreq]
    vpred = {col.index(k):pred_req[i] for i,k in enumerate(features)}
    vvpred = conv.Convert_Req(vpred)
    dif = list(set(col).difference(set(features)))
    X_train = df[features]
    y_train = df[dif[0]]
    
    # print(features)
    # print(vvpred)

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train,y_train)
    
    #tracer l'arbre
    data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('img\TpTree.png')
     
    img=pltimg.imread('img\TpTree.png')
    imgplot = plt.imshow(img)
    prediction = dtree.predict([vvpred])
    pr = conv.Reconvert(prediction[0],col.index(dif[0]))
    print(f"{dif[0]} : {pr}")
    plt.show()

df = pd.read_csv("dataSetAchatVoiture.csv")
req = input("entrer le predict : ")
prediction(df,req)
