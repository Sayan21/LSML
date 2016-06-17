from numpy import *;
from operator import itemgetter;
import heapq;
from sys import *;
import os;
import logging;
from sklearn.metrics import *;


path.append(os.path.abspath('/Users/sayantandasgupta/workspace/LEML/src'));
import arffio;

#150K: 987
#300K: 3427
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 1.0

    return score / len(actual)

#logging.basicConfig(filename=argv[4]+'/MAP_logfile.txt',level=logging.DEBUG)

Pz_u=loadtxt(argv[1],delimiter=" ");

Pl_z=loadtxt(argv[2],delimiter=" ");

nU=Pz_u.shape[1]; K= Pz_u.shape[0];  nL=Pl_z.shape[1];

print 'Number of Users: %d Number of Items: %d Number of Latent States: %d'% (nU,nL,K);
Pz_u=matrix(Pz_u); Pl_z = transpose(matrix(Pl_z));

reader  = arffio.SvmReader(argv[3], batch = 1000000000000);
x, y    = reader.full_read();

nTest = x.shape[0];

M = [1,2,5,10,20,50]; M_max=M[len(M)-1];


userCount=0; sumAUC=0; sumF1=0; sumRankLoss=0;
import time; t0=time.time(); sumAPScore=0;
    
    
for test in range(0,nTest):
        
    P=Pl_z*Pz_u[:,test];
    P=P/sum(P);
    
    _,cols = y[test,:].nonzero();
    actual = cols.tolist();
    
    if actual:
        userCount+=1; 
        sumAUC+=roc_auc_score(y[test,:].toarray()[0],P);
        
        #for i in heapq.nlargest(len(actual), enumerate(P.A1.tolist()), key=lambda x:x[1]): pred[i[0]]=1;
        
        if(userCount%1000==0): 
            print 'Number of Documents %d, AUC %f' % (userCount,sumAUC/userCount)

print 'Number of Documents %d, AUC %f' % (userCount,sumAUC/userCount)
