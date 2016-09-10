# -*- coding: utf-8 -*-
def classify(features_train, labels_train):   
    # 1. importe o módulo GaussianNB da biblioteca sklearn
    # 2. crie o classificador
    # 3. utilize os atributos e rótulos de treinamento para treinar o classificador
    # 4. devolva o classificador treinado
    
    from sklearn.naive_bayes import GaussianNB
    ### create classifier
    clf = GaussianNB()
    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    ### return the fit classifier
    return clf
