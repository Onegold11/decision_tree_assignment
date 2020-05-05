from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
import pydot
import pandas as pd
import numpy as np

mushroom_data = pd.read_csv("C:/Users/ujini/Desktop/mushrooms.csv")

mushroom_data["class"] = mushroom_data["class"].replace('p', 0)
mushroom_data["class"] = mushroom_data["class"].replace('e', 1)

mushroom_data["cap-shape"] = mushroom_data["cap-shape"].replace("b", 0)
mushroom_data["cap-shape"] = mushroom_data["cap-shape"].replace("c", 1)
mushroom_data["cap-shape"] = mushroom_data["cap-shape"].replace("f", 2)
mushroom_data["cap-shape"] = mushroom_data["cap-shape"].replace("k", 3)
mushroom_data["cap-shape"] = mushroom_data["cap-shape"].replace("s", 4)
mushroom_data["cap-shape"] = mushroom_data["cap-shape"].replace("x", 5)

mushroom_data["cap-surface"] = mushroom_data["cap-surface"].replace("f", 0)
mushroom_data["cap-surface"] = mushroom_data["cap-surface"].replace("g", 1)
mushroom_data["cap-surface"] = mushroom_data["cap-surface"].replace("s", 2)
mushroom_data["cap-surface"] = mushroom_data["cap-surface"].replace("y", 3)

mushroom_data["cap-color"] = mushroom_data["cap-color"].replace("b", 0)
mushroom_data["cap-color"] = mushroom_data["cap-color"].replace("c", 1)
mushroom_data["cap-color"] = mushroom_data["cap-color"].replace("e", 2)
mushroom_data["cap-color"] = mushroom_data["cap-color"].replace("g", 3)
mushroom_data["cap-color"] = mushroom_data["cap-color"].replace("n", 4)
mushroom_data["cap-color"] = mushroom_data["cap-color"].replace("p", 5)
mushroom_data["cap-color"] = mushroom_data["cap-color"].replace("r", 6)
mushroom_data["cap-color"] = mushroom_data["cap-color"].replace("u", 7)
mushroom_data["cap-color"] = mushroom_data["cap-color"].replace("w", 8)
mushroom_data["cap-color"] = mushroom_data["cap-color"].replace("y", 9)

mushroom_data["bruises"] = mushroom_data["bruises"].replace("f", 0)
mushroom_data["bruises"] = mushroom_data["bruises"].replace("t", 1)

mushroom_data["odor"] = mushroom_data["odor"].replace("a", 0)
mushroom_data["odor"] = mushroom_data["odor"].replace("c", 1)
mushroom_data["odor"] = mushroom_data["odor"].replace("f", 2)
mushroom_data["odor"] = mushroom_data["odor"].replace("l", 3)
mushroom_data["odor"] = mushroom_data["odor"].replace("m", 4)
mushroom_data["odor"] = mushroom_data["odor"].replace("n", 5)
mushroom_data["odor"] = mushroom_data["odor"].replace("p", 6)
mushroom_data["odor"] = mushroom_data["odor"].replace("s", 7)
mushroom_data["odor"] = mushroom_data["odor"].replace("y", 8)

mushroom_data["gill-attachment"] = mushroom_data["gill-attachment"].replace("a", 0)
mushroom_data["gill-attachment"] = mushroom_data["gill-attachment"].replace("f", 1)

mushroom_data["gill-spacing"] = mushroom_data["gill-spacing"].replace("c", 0)
mushroom_data["gill-spacing"] = mushroom_data["gill-spacing"].replace("w", 1)

mushroom_data["gill-size"] = mushroom_data["gill-size"].replace("b", 0)
mushroom_data["gill-size"] = mushroom_data["gill-size"].replace("n", 1)

mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("b", 0)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("e", 1)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("g", 2)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("h", 3)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("k", 4)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("n", 5)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("o", 6)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("p", 7)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("r", 8)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("u", 9)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("w", 10)
mushroom_data["gill-color"] = mushroom_data["gill-color"].replace("y", 11)

mushroom_data["stalk-shape"] = mushroom_data["stalk-shape"].replace("e", 0)
mushroom_data["stalk-shape"] = mushroom_data["stalk-shape"].replace("t", 1)

mushroom_data["stalk-root"] = mushroom_data["stalk-root"].replace("?", 0)
mushroom_data["stalk-root"] = mushroom_data["stalk-root"].replace("b", 1)
mushroom_data["stalk-root"] = mushroom_data["stalk-root"].replace("c", 2)
mushroom_data["stalk-root"] = mushroom_data["stalk-root"].replace("e", 3)
mushroom_data["stalk-root"] = mushroom_data["stalk-root"].replace("r", 4)

mushroom_data["stalk-surface-above-ring"] = mushroom_data["stalk-surface-above-ring"].replace("f", 0)
mushroom_data["stalk-surface-above-ring"] = mushroom_data["stalk-surface-above-ring"].replace("k", 1)
mushroom_data["stalk-surface-above-ring"] = mushroom_data["stalk-surface-above-ring"].replace("s", 2)
mushroom_data["stalk-surface-above-ring"] = mushroom_data["stalk-surface-above-ring"].replace("y", 3)

mushroom_data["stalk-color-above-ring"] = mushroom_data["stalk-color-above-ring"].replace("b", 0)
mushroom_data["stalk-color-above-ring"] = mushroom_data["stalk-color-above-ring"].replace("c", 1)
mushroom_data["stalk-color-above-ring"] = mushroom_data["stalk-color-above-ring"].replace("e", 2)
mushroom_data["stalk-color-above-ring"] = mushroom_data["stalk-color-above-ring"].replace("g", 3)
mushroom_data["stalk-color-above-ring"] = mushroom_data["stalk-color-above-ring"].replace("n", 4)
mushroom_data["stalk-color-above-ring"] = mushroom_data["stalk-color-above-ring"].replace("o", 5)
mushroom_data["stalk-color-above-ring"] = mushroom_data["stalk-color-above-ring"].replace("p", 6)
mushroom_data["stalk-color-above-ring"] = mushroom_data["stalk-color-above-ring"].replace("w", 7)
mushroom_data["stalk-color-above-ring"] = mushroom_data["stalk-color-above-ring"].replace("y", 8)

mushroom_data["stalk-color-below-ring"] = mushroom_data["stalk-color-below-ring"].replace("b", 0)
mushroom_data["stalk-color-below-ring"] = mushroom_data["stalk-color-below-ring"].replace("c", 1)
mushroom_data["stalk-color-below-ring"] = mushroom_data["stalk-color-below-ring"].replace("e", 2)
mushroom_data["stalk-color-below-ring"] = mushroom_data["stalk-color-below-ring"].replace("g", 3)
mushroom_data["stalk-color-below-ring"] = mushroom_data["stalk-color-below-ring"].replace("n", 4)
mushroom_data["stalk-color-below-ring"] = mushroom_data["stalk-color-below-ring"].replace("o", 5)
mushroom_data["stalk-color-below-ring"] = mushroom_data["stalk-color-below-ring"].replace("p", 6)
mushroom_data["stalk-color-below-ring"] = mushroom_data["stalk-color-below-ring"].replace("w", 7)
mushroom_data["stalk-color-below-ring"] = mushroom_data["stalk-color-below-ring"].replace("y", 8)

mushroom_data["veil-type"] = mushroom_data["veil-type"].replace("p", 0)

mushroom_data["veil-color"] = mushroom_data["veil-color"].replace("n", 0)
mushroom_data["veil-color"] = mushroom_data["veil-color"].replace("o", 1)
mushroom_data["veil-color"] = mushroom_data["veil-color"].replace("w", 2)
mushroom_data["veil-color"] = mushroom_data["veil-color"].replace("y", 3)

mushroom_data["ring-number"] = mushroom_data["ring-number"].replace("n", 0)
mushroom_data["ring-number"] = mushroom_data["ring-number"].replace("o", 1)
mushroom_data["ring-number"] = mushroom_data["ring-number"].replace("t", 2)

mushroom_data["ring-type"] = mushroom_data["ring-type"].replace("e", 0)
mushroom_data["ring-type"] = mushroom_data["ring-type"].replace("f", 1)
mushroom_data["ring-type"] = mushroom_data["ring-type"].replace("l", 2)
mushroom_data["ring-type"] = mushroom_data["ring-type"].replace("n", 3)
mushroom_data["ring-type"] = mushroom_data["ring-type"].replace("p", 4)

mushroom_data["spore-print-color"] = mushroom_data["spore-print-color"].replace("b", 0)
mushroom_data["spore-print-color"] = mushroom_data["spore-print-color"].replace("h", 1)
mushroom_data["spore-print-color"] = mushroom_data["spore-print-color"].replace("k", 2)
mushroom_data["spore-print-color"] = mushroom_data["spore-print-color"].replace("n", 3)
mushroom_data["spore-print-color"] = mushroom_data["spore-print-color"].replace("o", 4)
mushroom_data["spore-print-color"] = mushroom_data["spore-print-color"].replace("r", 5)
mushroom_data["spore-print-color"] = mushroom_data["spore-print-color"].replace("u", 6)
mushroom_data["spore-print-color"] = mushroom_data["spore-print-color"].replace("w", 7)
mushroom_data["spore-print-color"] = mushroom_data["spore-print-color"].replace("y", 8)

mushroom_data["population"] = mushroom_data["population"].replace("a", 0)
mushroom_data["population"] = mushroom_data["population"].replace("c", 1)
mushroom_data["population"] = mushroom_data["population"].replace("n", 2)
mushroom_data["population"] = mushroom_data["population"].replace("s", 3)
mushroom_data["population"] = mushroom_data["population"].replace("v", 4)
mushroom_data["population"] = mushroom_data["population"].replace("y", 5)

mushroom_data["habitat"] = mushroom_data["habitat"].replace("d", 0)
mushroom_data["habitat"] = mushroom_data["habitat"].replace("g", 1)
mushroom_data["habitat"] = mushroom_data["habitat"].replace("l", 2)
mushroom_data["habitat"] = mushroom_data["habitat"].replace("m", 3)
mushroom_data["habitat"] = mushroom_data["habitat"].replace("p", 4)
mushroom_data["habitat"] = mushroom_data["habitat"].replace("u", 5)
mushroom_data["habitat"] = mushroom_data["habitat"].replace("w", 6)

X = np.array(pd.DataFrame(mushroom_data, columns=["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]))
y = np.array(pd.DataFrame(mushroom_data, columns=['class']))

X_train, X_test, y_train, y_test = train_test_split(X, y)

#의사결정 트리 선언(criterion: 분류 기준 값(entropy, gini), max_depth: 최대 깊이)
dTreeMushroom1 = DecisionTreeClassifier(criterion="gini", max_depth=5)
dTreeMushroom1.fit(X_train, y_train)

dTreeMushroom2 = DecisionTreeClassifier(criterion="gini", max_depth=3)
dTreeMushroom2.fit(X_train, y_train)

dTreeMushroom3 = DecisionTreeClassifier(criterion="entropy", max_depth=5)
dTreeMushroom3.fit(X_train, y_train)

dTreeMushroom4 = DecisionTreeClassifier(criterion="entropy", max_depth=3)
dTreeMushroom4.fit(X_train, y_train)

y_test_predict1 = dTreeMushroom1.predict(X_test)
print("accuracy(1) : %f" % accuracy_score(y_test, y_test_predict1))

y_test_predict2 = dTreeMushroom2.predict(X_test)
print("accuracy(2) : %f" % accuracy_score(y_test, y_test_predict2))

y_test_predict3 = dTreeMushroom3.predict(X_test)
print("accuracy(3) : %f" % accuracy_score(y_test, y_test_predict3))

y_test_predict4 = dTreeMushroom4.predict(X_test)
print("accuracy(4) : %f" % accuracy_score(y_test, y_test_predict4))

export_graphviz(dTreeMushroom1, out_file="dTreeMushroom1.dot", class_names=['Poisonous', 'Edible'],
                feature_names=["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"], impurity=True, filled=True)

export_graphviz(dTreeMushroom2, out_file="dTreeMushroom2.dot", class_names=['Poisonous', 'Edible'],
                feature_names=["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"], impurity=True, filled=True)

export_graphviz(dTreeMushroom3, out_file="dTreeMushroom3.dot", class_names=['Poisonous', 'Edible'],
                feature_names=["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"], impurity=True, filled=True)

export_graphviz(dTreeMushroom4, out_file="dTreeMushroom4.dot", class_names=['Poisonous', 'Edible'],
                feature_names=["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"], impurity=True, filled=True)
#Encoding 중요
(graph1,) = pydot.graph_from_dot_file('dTreeMushroom1.dot', encoding="utf-8")
(graph2,) = pydot.graph_from_dot_file('dTreeMushroom2.dot', encoding="utf-8")
(graph3,) = pydot.graph_from_dot_file('dTreeMushroom3.dot', encoding="utf-8")
(graph4,) = pydot.graph_from_dot_file('dTreeMushroom4.dot', encoding="utf-8")

#Dot 파일을 Png 이미지로 저장
graph1.write_png('dTreeMushroom1.png')
graph2.write_png('dTreeMushroom2.png')
graph3.write_png('dTreeMushroom3.png')
graph4.write_png('dTreeMushroom4.png')