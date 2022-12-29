import pandas as pd 
import itertools

def eq(c, attr):
    #print(c, attr)
    flag = False
    for a in attr:
        if(c == a):
            #print(":)")
            flag = True
            break
    return flag

def Gini(df, class_name, attr1, attr2):
    attrs1 = ""
    for a in attr1:
        attrs1 = attrs1 + "\'" + a[1:] + "\',"
    attrs1 = attrs1[:-1]
    attrs2 = ""
    for a in attr2:
        attrs2 = attrs2 + "\'" + a[1:] + "\',"
    attrs2 = attrs2[:-1]
    
    #print(intro)
    n1_0 = 0
    n1_1 = 0
    n2_0 = 0
    n2_1 = 0
    #print(df[class_name][0])
    #print(attr1)
    for i in df.index:
        if(eq(df[class_name][i], attr1)):
            #print(i)
            if(df[" Class"][i] == " C0"):
                n1_0 += 1
            else: 
                n1_1 += 1
        else:
            if(df[" Class"][i] == " C0"):
                n2_0 += 1
            else: 
                n2_1 += 1
    n1 = n1_0 + n1_1
    n2 = n2_0 + n2_1
    alll = n1 + n2
    if(n1 != 0):
        gini_n1 = 1 - (n1_0/n1)**2 - (n1_1/n1)**2
    else:
        gini_n1 = 1
    if(n2 != 0):
        gini_n2 = 1 - (n2_0/n2)**2 - (n2_1/n2)**2
    else:
        gini_n2 = 1
    gini = n1/alll*gini_n1 + n2/alll*gini_n2
    intro = (class_name + ": {" + attrs1 + "}" +" {" + attrs2 + "}"+ " -> gini = "+str(gini))[1:]
    print(intro)

if __name__ == "__main__":  
    path = "data.csv"   
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    print("======= step1 =======")
    Gini(df, " Gender", [' M'], [' F'])
    Gini(df, " Car Type", [' Family'], [' Sports',' Luxury'])
    Gini(df, " Car Type", [' Sports'], [' Family',' Luxury'])
    Gini(df, " Car Type", [' Luxury'], [' Family',' Sports'])
    Gini(df, " Shirt Size", [' Small'], [' Medium',' Large', ' Extra Large'])
    Gini(df, " Shirt Size", [' Medium'], [' Small',' Large', ' Extra Large'])
    Gini(df, " Shirt Size", [' Large'], [' Small',' Medium', ' Extra Large'])
    Gini(df, " Shirt Size", [' Extra Large'], [' Small',' Medium', ' Large'])
    Gini(df, " Shirt Size", [' Small',' Medium'], [' Large', ' Extra Large'])
    Gini(df, " Shirt Size", [' Small', ' Large'], [' Medium', ' Extra Large'])
    Gini(df, " Shirt Size", [' Small', ' Extra Large'], [' Medium', ' Large'])
    print("---> min gini = 0.27692307692307694")
    print("\n======= step2 =======")
    df2_1 = df[df[' Car Type'] == " Family"] 
    #print(df2_1)
    #print(df2_1)
    Gini(df2_1, " Gender", [' M'], [' F'])
    Gini(df2_1, " Shirt Size", [' Small'], [' Medium',' Large', ' Extra Large'])
    Gini(df2_1, " Shirt Size", [' Medium'], [' Small',' Large', ' Extra Large'])
    Gini(df2_1, " Shirt Size", [' Large'], [' Small',' Medium', ' Extra Large'])
    Gini(df2_1, " Shirt Size", [' Extra Large'], [' Small',' Medium', ' Large'])
    Gini(df2_1, " Shirt Size", [' Small',' Medium'], [' Large', ' Extra Large'])
    Gini(df2_1, " Shirt Size", [' Small', ' Large'], [' Medium', ' Extra Large'])
    Gini(df2_1, " Shirt Size", [' Small', ' Extra Large'], [' Medium', ' Large'])
    print("---> min gini = 0.0")
    print("leaf node: C0")


    print("\n======= step3 =======")

    df3 = df[df[' Car Type'].isin([' Sports', ' Luxury'])] 
    Gini(df3, " Gender", [' M'], [' F'])
    Gini(df3, " Car Type", [' Sports'], [' Luxury'])
    Gini(df3, " Shirt Size", [' Small'], [' Medium',' Large', ' Extra Large'])
    Gini(df3, " Shirt Size", [' Medium'], [' Small',' Large', ' Extra Large'])
    Gini(df3, " Shirt Size", [' Large'], [' Small',' Medium', ' Extra Large'])
    Gini(df3, " Shirt Size", [' Extra Large'], [' Small',' Medium', ' Large'])
    Gini(df3, " Shirt Size", [' Small',' Medium'], [' Large', ' Extra Large'])
    Gini(df3, " Shirt Size", [' Small', ' Large'], [' Medium', ' Extra Large'])
    Gini(df3, " Shirt Size", [' Small', ' Extra Large'], [' Medium', ' Large'])
    print("---> min gini = 0.12307692307692303")

    print("\n======= step4 =======")
    df4 = df3[df3[' Shirt Size'].isin([' Small', ' Medium'])] 
    Gini(df4, " Gender", [' M'], [' F'])
    Gini(df4, " Car Type", [' Sports'], [' Luxury'])
    Gini(df4, " Shirt Size", [' Small'], [' Medium'])
    print("leaf node: C1")

    print("\n======= step5 =======")
    #print(df3)
    
    df5 = df3[df3[' Shirt Size'].isin([' Large', ' Extra Large'])] 
    #print(df5)
    Gini(df5, " Gender", [' M'], [' F'])
    Gini(df5, " Car Type", [' Sports'], [' Luxury'])
    Gini(df5, " Shirt Size", [' Large'], [' Extra Large'])
    print("---> min gini = 0.2")

    print("\n======= step6 =======")
    df6 = df5[df5[' Car Type'].isin([' Sports'])] 
    Gini(df6, " Gender", [' M'], [' F'])
    Gini(df6, " Shirt Size", [' Large'], [' Extra Large'])
    print("---> min gini = 0.0")

    print("\n======= step7 =======")
    df7 = df5[df5[' Car Type'].isin([' Luxury'])] 
    #print(df7)
    Gini(df7, " Gender", [' M'], [' F'])
    Gini(df7, " Shirt Size", [' Large'], [' Extra Large'])
    print("---> min gini = 0.0")    
    print("leaf node: C0")

    print("\n======= step8 =======")
    df8 = df6[df6[' Shirt Size'].isin([' Large'])] 
    print("leaf node: C1")

    print("\n======= step9 =======")
    df9 = df6[df6[' Shirt Size'].isin([' Extra Large'])] 
    print("leaf node: C0")
    