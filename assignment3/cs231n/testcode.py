import numpy as np
a=np.array([1,2,3,4])
b=np.array([1,2])
np.add.at(a,[0,1],b) 
print(a)

dW=np.zeros((5,6),int)
np.add.at(dW,[[1,2],[2,4,4]],[[1],[2]]) #1,2 +1; 1,4+1 1,4+1  2,2+2  2，4+2 2,4+2 
print(dW)