from bertopic import BERTopic

loaded_model = BERTopic.load("final-20")
print(loaded_model.get_topic_info())

pdf = loaded_model.get_topic_info()
for i in range(len(pdf)):
    print(pdf.iloc[i])
    print()

sc = 0
for i in range(len(pdf)):
    print(pdf.iloc[i]["Representation"], pdf.iloc[i]["Count"])
    print()
    sc += pdf.iloc[i]["Count"]
