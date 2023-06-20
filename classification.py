# features (1 sim, 0 não)
# longo?
# perna curta?
# faz auau?
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 => porco, 0 => cachorro
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0] # labels / etiqueta

model = LinearSVC()
model.fit(treino_x, treino_y)

animal_misterioso = [1, 1, 1]
print(model.predict([animal_misterioso]))

misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]

teste_x = [misterio1, misterio2, misterio3]
teste_y = [0, 1, 1]

previsoes = model.predict(teste_x)

corretos = (previsoes == teste_y).sum()
total = len(teste_x)
taxa_de_acerto = corretos / total
print("Taxa de acerto: ", taxa_de_acerto * 100)

taxa_de_acerto = accuracy_score(teste_y, previsoes)
print("Taxa de acerto", taxa_de_acerto * 100)
