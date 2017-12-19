from sklearn import tree
import numpy as np
from sklearn.datasets import load_iris
iris=load_iris()
print iris.feature_names
print iris.target_names
test_idx=[0,50,100]
train_target=np.delete(iris.target,test_idx)
train_data=np.delete(iris.data,test_idx,axis=0)

test_target=iris.target[test_idx]
test_data=iris.data[test_idx]

clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print test_target
print clf.predict(test_data)




import graphviz 
import pydot
from sklearn.externals.six import StringIO
dot_data=StringIO()
dot_data = tree.export_graphviz(clf, out_file=dot_data, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                        impurity=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf("iris.pdf")
