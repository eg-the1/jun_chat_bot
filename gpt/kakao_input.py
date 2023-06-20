text = open("C://Users//kds96//OneDrive//바탕 화면//PYTHON//gpt//KakaoTalkChats.txt", 'r',encoding="utf-8-sig")
a = text.readlines()
chatbot_data=[]
chatbot=[i.strip().split(",") for i in a if i.strip() != ""]
for i in chatbot:
    if len(i) == 1:
        pass
    else:
        split_result = i[1].split(':')
        if len(split_result) == 2:
            a, b = split_result
            chatbot_data.append([a.strip(),b.strip()])
try:
    for i in range(len(chatbot_data)):
        if chatbot_data[i][0]=='pass':
            continue
        if chatbot_data[i][0]=='김준서':
            state = True
            b = 0
            context=chatbot_data[i][1]
            while state:
                b += 1
                if chatbot_data[i+b][0]=='김준서':
                    context+=" "+chatbot_data[i+b][1]
                    chatbot_data[i+b][0]="pass"
                    chatbot_data[i+b][1]="pass"
                else:
                    state = False
            chatbot_data[i][1]=context
        if chatbot_data[i][0]=='세준':
            state = True
            b = 0
            context=chatbot_data[i][1]
            while state:
                b += 1
                if chatbot_data[i+b][0]=='세준':
                    context+=" "+chatbot_data[i+b][1]
                    chatbot_data[i+b][0]="pass"
                    chatbot_data[i+b][1]="pass"
                else:
                    state = False
            chatbot_data[i][1]=context
except:
    pass
chatbot_data = [i for i in chatbot_data if i[0] != "pass"]
question = []
answer= []
for i in range(len(chatbot_data)-1):
    question.append(chatbot_data[i][1]) 
    answer.append(chatbot_data[i+1][1])
# for i in range(len(chatbot_data)-1):
#     print(f"1{question[i]}2{answer[i]}")
print(len(question))
