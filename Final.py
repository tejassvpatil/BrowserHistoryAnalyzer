import browserhistory as bh
import csv
import pandas as pd

dict_obj = bh.get_browserhistory()

b = bh.get_database_paths()
print(b)

a = list(dict_obj.items())
l2 = a[0]
l3 = l2[1]

df = pd.DataFrame(l3,columns = ['Url','Name','Timestamp'])
new_df = pd.DataFrame(df['Timestamp'].str.split(' ',1).tolist(),columns = ['Date' , 'Time'])
df = df.drop(columns = ['Timestamp'])
df3 = pd.concat([df,new_df],axis = 1,sort = False)

df3.to_csv("./browser_history.csv", sep =',', index = False)

def least5():
 import numpy as np
 import matplotlib.pyplot as plt
 import csv
 import pandas as pd
 from urllib.parse import urlparse
 from collections import defaultdict
 import datetime
 from dateutil.relativedelta import relativedelta   

 df = pd.read_csv('browser_history.csv') 

 c = []

 for i in df['Url']:
      k = urlparse(i)
      a = k.netloc
      c.append(a)

 appearances = defaultdict(int)
 fig = plt.figure()

 for curr in c:
     appearances[curr] += 1

 def takeSecond(elem):
     return elem[1]
 
 adk = list(appearances.items())
 adk.sort(key = takeSecond, reverse = True)
 df1 = pd.DataFrame(adk,columns = ['Url', 'Count'])
 df1 = df1[df1.Url != '']
 df1.to_csv('ucl.csv', sep = ',', index = False)
 print(df1.head(5))


 tail1=np.array(df1['Count'].tail(5))


 tail2=np.array(df1['Url'].tail(5))
 fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))

 def func(pct, allvals):
     absolute = int(pct/100.*np.sum(allvals))
     return "{:.1f}%\n({:d} )".format(pct, absolute)


 wedges, texts, autotexts = ax.pie(tail1, autopct=lambda pct: func(pct, tail1),
                                   textprops=dict(color="w"))

 ax.legend(wedges, tail2,
           title="Websites",
           loc="center left",
           bbox_to_anchor=(1, 0, 0.5, 1))

 plt.setp(autotexts, size=8, weight="bold")

 ax.set_title("Least 5 visited websites")


 fig.savefig("L5.png")
 def todayleast5():
        
        import matplotlib.pyplot as plt
        import pandas as pd

        import numpy as np

        fig=plt.figure()


        df = pd.read_csv('todaysh.csv')
        # x-coordinates of left sides of bars  
        left = df['Url'].tail(5)
        
        # heights of bars 
        height = df['Count'].tail(5)
        
        # labels for bars 
        tick_label = df['Url'].tail(5)
        plt.xticks(np.arange(1,5), df['Url'].tail(5), rotation=90)
        # plotting a bar chart 
        plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = 'red') 
        
        # naming the x-axis 
        plt.xlabel('URL') 
        # naming the y-axis 
        plt.ylabel('Count') 
        # plot title 
        plt.title('Todays Least 5') 

        fig.savefig("todayleast5.png")
 def lastweekleast5():
        import matplotlib.pyplot as plt
        import pandas as pd

        import numpy as np

        fig=plt.figure()
        df = pd.read_csv('lweeksh.csv')
        # x-coordinates of left sides of bars  
        left = df['Url'].tail(5)
        
        # heights of bars 
        height = df['Count'].tail(5)
        
        # labels for bars 
        tick_label = df['Url'].tail(5)
        plt.xticks(np.arange(1,5), df['Url'].tail(5), rotation=90)
        # plotting a bar chart 
        plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = 'red') 
        
        # naming the x-axis 
        plt.xlabel('URL') 
        # naming the y-axis 
        plt.ylabel('Count') 
        # plot title 
        plt.title('Last Weeks Least 5') 
      
        # function to show the plot 

        fig.savefig("lwleast5.png")
 
 lastweekleast5()
 todayleast5()

def top5():
    import numpy as np
    import matplotlib.pyplot as plt
    import csv
    import pandas as pd
    from urllib.parse import urlparse
    from collections import defaultdict
    import datetime
    from dateutil.relativedelta import relativedelta

    df = pd.read_csv('browser_history.csv')

    c = []

    for i in df['Url']:
        k = urlparse(i)
        a = k.netloc
        c.append(a)

    appearances = defaultdict(int)

    for curr in c:
        appearances[curr] += 1

    def takeSecond(elem):
        return elem[1]

    adk = list(appearances.items())
    adk.sort(key = takeSecond, reverse = True)
    df1 = pd.DataFrame(adk,columns = ['Url', 'Count'])
    df1 = df1[df1.Url != '']
    df1.to_csv('ucl.csv', sep = ',', index = False)
    print(df1.head(5))

    arr=np.array(df1['Count'].head(5))


    arr1=np.array(df1['Url'].head(5))

    

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d} )".format(pct, absolute)


    wedges, texts, autotexts = ax.pie(arr, autopct=lambda pct: func(pct, arr),
                                    textprops=dict(color="w"))

    ax.legend(wedges, arr1,
            title="Websites",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")

    ax.set_title("Top 5 visited websites")
    fig.savefig("T5.png")

    def todaytop5():
            import matplotlib.pyplot as plt
            import pandas as pd

            import numpy as np

            fig=plt.figure()
            df = pd.read_csv('todaysh.csv')
            # x-coordinates of left sides of bars  
            left = df['Url'].head(5)
            
            # heights of bars 
            height = df['Count'].head(5)
            
            # labels for bars 
            tick_label = df['Url'].head(5)
            plt.xticks(np.arange(1,5), df['Url'].head(5), rotation=90)
            # plotting a bar chart 
            plt.bar(left, height, tick_label = tick_label, 
                    width = 0.8, color = 'red') 

                
            
            # naming the x-axis 
            plt.xlabel('URL') 
            # naming the y-axis 
            plt.ylabel('Count') 
            # plot title 
            plt.title('Todays Top 5') 

            from matplotlib.pyplot import figure

            figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
            
            # function to show the plot 
            fig.savefig("todaytop5.png" ,aspect='auto')

    def lastweektop5():
        import matplotlib.pyplot as plt
        import pandas as pd

        import numpy as np

        fig=plt.figure()
        df = pd.read_csv('lweeksh.csv')
        # x-coordinates of left sides of bars  
        left = df['Url'].head(5)
        
        # heights of bars 
        height = df['Count'].head(5)
        
        # labels for bars 
        tick_label = df['Url'].head(5)
        plt.xticks(np.arange(1,5), df['Url'].head(5), rotation=90)
        # plotting a bar chart 
        plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = 'red') 
        
        # naming the x-axis 
        plt.xlabel('URL') 
        # naming the y-axis 
        plt.ylabel('Count') 
        # plot title 
        plt.title('Last Weeks top 5') 
        fig.savefig("lwtop5.png")        
    lastweektop5()
    todaytop5()


def time():
    import numpy as np
    import matplotlib as mpl
    import csv
    import pandas as pd
    from urllib.parse import urlparse
    from collections import defaultdict
    import datetime as dt
    from datetime import time,datetime
    from dateutil.relativedelta import relativedelta

    today = dt.date.today()
    lweek = today + relativedelta(days=-7)
    first = today.replace(day = 1)
    lmonth = first + relativedelta(months=-1)
    lsan = first + relativedelta(months=-6)
    lyear = first + relativedelta(years=-1)
    #lmonth = first - datetime.timedelta(days=30)

    today = today.strftime("%Y-%m-%d")
    first = first.strftime("%Y-%m-%d")
    lweek = lweek.strftime("%Y-%m-%d")
    lmonth = lmonth.strftime("%Y-%m-%d")
    lsan = lsan.strftime("%Y-%m-%d")
    lyear = lyear.strftime("%Y-%m-%d")

    today = datetime.strptime(today,"%Y-%m-%d")
    first = datetime.strptime(first,"%Y-%m-%d")
    lweek = datetime.strptime(lweek,"%Y-%m-%d")
    lmonth = datetime.strptime(lmonth,"%Y-%m-%d")
    lsan = datetime.strptime(lsan,"%Y-%m-%d")
    lyear = datetime.strptime(lyear,"%Y-%m-%d")

    today = today.date()
    first = first.date()
    lweek = lweek.date()
    lmonth = lmonth.date()
    lsan = lsan.date()
    lyear = lyear.date()

    print(today)
    print(lweek)
    print(first)
    print(lmonth)
    print(lsan)
    print(lyear)

    df = pd.read_csv('browser_history.csv')

    ct = [];cw = [];cm = [];cs = [];cy = [];ca = []

    for f,i in zip(df['Date'],df['Url']):
        f1 = datetime.strptime(f, '%Y-%m-%d')
        f1 = f1.date()
        if (f1 == today):
            k = urlparse(i)
            a = k.netloc
            ct.append(a)

        if (f1 >= lweek and f1 <= today):
            k = urlparse(i)
            a = k.netloc
            cw.append(a)

        if (f1 >= lmonth and f1 <= today):
            k = urlparse(i)
            a = k.netloc
            cm.append(a)

        if (f1 >= lsan and f1 <= today):
            k = urlparse(i)
            a = k.netloc
            cs.append(a)

        if (f1 >= lyear and f1 <= today):
            k = urlparse(i)
            a = k.netloc
            cy.append(a)

            k = urlparse(i)
            a = k.netloc
            ca.append(a)

    def takeSecond(elem):
        return elem[1]

    app1 = defaultdict(int)
    for curr in ct:
        app1[curr] += 1

    adk = list(app1.items())
    adk.sort(key = takeSecond, reverse = True)
    df1 = pd.DataFrame(adk,columns = ['Url', 'Count'])
    df1 = df1[df1.Url != '']
    df1.to_csv('todaysh.csv', sep = ',', index = False)

    ###################################################

    app2 = defaultdict(int)
    for curr in cw:
        app2[curr] += 1

    adk2 = list(app2.items())
    adk2.sort(key = takeSecond, reverse = True)
    df2 = pd.DataFrame(adk2,columns = ['Url', 'Count'])
    df2 = df2[df2.Url != '']
    df2.to_csv('lweeksh.csv', sep = ',', index = False)
    #print(df2.head(5))

    ###########################################

    app3 = defaultdict(int)
    for curr in cm:
        app3[curr] += 1

    adk3 = list(app3.items())
    adk3.sort(key = takeSecond, reverse = True)
    df3 = pd.DataFrame(adk3,columns = ['Url', 'Count'])
    df3 = df3[df3.Url != '']
    df3.to_csv('lmonthsh.csv', sep = ',', index = False)

    #################################

    app4 = defaultdict(int)
    for curr in cs:
        app4[curr] += 1

    adk4 = list(app4.items())
    adk4.sort(key = takeSecond, reverse = True)
    df4 = pd.DataFrame(adk4,columns = ['Url', 'Count'])
    df4 = df4[df4.Url != '']
    df4.to_csv('lsixh.csv', sep = ',', index = False)

    #################################

    app5 = defaultdict(int)
    for curr in cs:
        app5[curr] += 1

    adk5 = list(app5.items())
    adk5.sort(key = takeSecond, reverse = True)
    df5 = pd.DataFrame(adk5,columns = ['Url', 'Count'])
    df5 = df5[df5.Url != '']
    df5.to_csv('lyearh.csv', sep = ',', index = False)

    #################################

    app6 = defaultdict(int)
    for curr in cs:
        app6[curr] += 1

    adk6 = list(app6.items())
    adk6.sort(key = takeSecond, reverse = True)
    df6 = pd.DataFrame(adk6,columns = ['Url', 'Count'])
    df6 = df6[df6.Url != '']
    df6.to_csv('lallh.csv', sep = ',', index = False)

    #################################

def busytop():

 def  busyweek():
    import numpy as np
    import matplotlib as mpl
    import csv
    import pandas as pd
    from urllib.parse import urlparse
    from collections import defaultdict
    import datetime as dt
    from datetime import time,datetime
    from dateutil.relativedelta import relativedelta

    df3 = pd.read_csv('browser_history.csv')

    today = dt.date.today()
    lweek = today + relativedelta(days = -7)
    today = today.strftime("%Y-%m-%d")
    lweek = lweek.strftime("%Y-%m-%d")
    print(today)
    print(lweek)

    c = []
    count0,count1,count2,count3,count4,count5,count6,count7 = 0,0,0,0,0,0,0,0

    time0 = time(0,0,0)
    time1 = time(1,0,0)
    time2 = time(2,0,0)
    time3 = time(3,0,0)
    time4 = time(4,0,0)
    time5 = time(5,0,0)
    time6 = time(6,0,0)
    time7 = time(7,0,0)
    time8 = time(8,0,0)
    time9 = time(9,0,0)
    time10 = time(10,0,0)
    time11 = time(11,0,0)
    time12 = time(12,0,0)
    time13 = time(13,0,0)
    time14 = time(14,0,0)
    time15 = time(15,0,0)
    time16 = time(16,0,0)
    time17 = time(17,0,0)
    time18 = time(18,0,0)
    time19 = time(19,0,0)
    time20 = time(20,0,0)
    time21 = time(21,0,0)
    time22 = time(22,0,0)
    time23 = time(23,0,0)

    for i,f,k in zip(df3['Time'],df3['Date'],df3['Url']):
        i1 = datetime.strptime(i, '%H:%M:%S')
        i1 = i1.time()
        if (f >= lweek):
            c.append(f)

    app1 = defaultdict(int)

    for curr in c:
        app1[curr] += 1

    def takeSecond(elem):
        return elem[1]

    def take1(elem):
        return elem[0]

    adk = list(app1.items())
    adk.sort(key = takeSecond, reverse = True)
    df1 = pd.DataFrame(adk,columns = ['Time', 'Count'])
    df1 = df1[df1.Time != '']
    df1.to_csv('lweekcount.csv', sep = ',', index = False)
    print(df1)
 busyweek()
 def busyweektop5():
      
        import matplotlib.pyplot as plt
        import pandas as pd

        import numpy as np

        fig=plt.figure()
        df = pd.read_csv('lweekcount.csv')
        # x-coordinates of left sides of bars  
        left = df['Time'].head(5)
        
        # heights of bars 
        height = df['Count'].head(5)
            

        

        # labels for bars 
        tick_label = df['Time'].head(5)
        plt.xticks(np.arange(1,5), df['Time'].head(5), rotation=30 , fontsize=8)
        # plotting a bar chart 
        plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = 'red')


        # naming the x-axis 
        plt.xlabel('Date') 
        # naming the y-axis 
        plt.ylabel('Count') 
        # plot title 
        plt.title('This Week top 5') 
        # function to show the plot  
        fig.savefig("bwtop5.png")
 busyweektop5()

 def busymonth():
    import numpy as np
    import matplotlib as mpl
    import csv
    import pandas as pd
    from urllib.parse import urlparse
    from collections import defaultdict
    import datetime as dt
    from datetime import time,datetime
    from dateutil.relativedelta import relativedelta

    df3 = pd.read_csv('browser_history.csv')

    today = dt.date.today()
    first = today.replace(day = 1)
    lweek = today + relativedelta(days = -7)
    lmonth = first + relativedelta(months = -1)
    today = today.strftime("%Y-%m-%d")
    lweek = lweek.strftime("%Y-%m-%d")
    lmonth = lmonth.strftime("%Y-%m-%d")
    print(today)
    print(lweek)
    print(lmonth)

    c = []

    for i,f,k in zip(df3['Time'],df3['Date'],df3['Url']):
        i1 = datetime.strptime(i, '%H:%M:%S')
        i1 = i1.time()
        if (f >= lmonth):
            c.append(f)

    app1 = defaultdict(int)

    for curr in c:
        app1[curr] += 1

    def takeSecond(elem):
        return elem[1]

    def take1(elem):
        return elem[0]

    adk = list(app1.items())
    adk.sort(key = takeSecond, reverse = True)
    df1 = pd.DataFrame(adk,columns = ['Time', 'Count'])
    df1 = df1[df1.Time != '']
    df1.to_csv('lmonthcount.csv', sep = ',', index = False)
    print(df1.head(5))
 busymonth()
 def busymonthtop5():
        
        import matplotlib.pyplot as plt
        import pandas as pd

        import numpy as np

        fig=plt.figure()
        df = pd.read_csv('lmonthcount.csv')
        # x-coordinates of left sides of bars  
        left = df['Time'].head(5)
        
        # heights of bars 
        height = df['Count'].head(5)
            

        

        # labels for bars 
        tick_label = df['Time'].head(5)
        plt.xticks(np.arange(1,5), df['Time'].head(5),rotation=30 , fontsize=8)
        # plotting a bar chart 
        plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = 'red')


        # naming the x-axis 
        plt.xlabel('URL') 
        # naming the y-axis 
        plt.ylabel('Count') 
        # plot title 
        plt.title('This month top 5') 
        # function to show the plot  
        fig.savefig("bmtop5.png")
 busymonthtop5()

 def busy6month():
    import numpy as np
    import matplotlib as mpl
    import csv
    import pandas as pd
    from urllib.parse import urlparse
    from collections import defaultdict
    import datetime as dt
    from datetime import time,datetime
    from dateutil.relativedelta import relativedelta

    df3 = pd.read_csv('browser_history.csv')

    today = dt.date.today()
    first = today.replace(day = 1)
    lweek = today + relativedelta(days = -7)
    lmonth = first + relativedelta(months = -1)
    lsan = first + relativedelta(months=-6)
    lyear = first + relativedelta(years=-1)

    today = today.strftime("%Y-%m-%d")
    lweek = lweek.strftime("%Y-%m-%d")
    lmonth = lmonth.strftime("%Y-%m-%d")
    lsan = lsan.strftime("%Y-%m-%d")
    lyear = lyear.strftime("%Y-%m-%d")

    print(today)
    print(lweek)
    print(lmonth)
    print(lsan)
    print(lyear)

    c = []

    for i,f,k in zip(df3['Time'],df3['Date'],df3['Url']):
        i1 = datetime.strptime(i, '%H:%M:%S')
        i1 = i1.time()
        if (f >= lsan):
            c.append(f)

    app1 = defaultdict(int)

    for curr in c:
        app1[curr] += 1

    def takeSecond(elem):
        return elem[1]

    def take1(elem):
        return elem[0]

    adk = list(app1.items())
    adk.sort(key = takeSecond, reverse = True)
    df1 = pd.DataFrame(adk,columns = ['Time', 'Count'])
    df1 = df1[df1.Time != '']
    df1.to_csv('lsancount.csv', sep = ',', index = False)
    print(df1.head(5))
 busy6month()
 def busy6monthtop5():

    import matplotlib.pyplot as plt
    import pandas as pd

    import numpy as np

    fig=plt.figure()
    df = pd.read_csv('lsancount.csv')
    # x-coordinates of left sides of bars  
    left = df['Time'].head(5)
    
    # heights of bars 
    height = df['Count'].head(5) 

    # labels for bars 
    tick_label = df['Time'].head(5)
    plt.xticks(np.arange(1,5), df['Time'].head(5),rotation=30 , fontsize=8)
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.8, color = 'red')


    # naming the x-axis 
    plt.xlabel('Date') 
    # naming the y-axis 
    plt.ylabel('Count') 
    # plot title 
    plt.title('Last six month busy days top 5') 
    # function to show the plot 
    
    fig.savefig("lsmbtop5.png")
 busy6monthtop5()

 def busyyear():
    import numpy as np
    import matplotlib as mpl
    import csv
    import pandas as pd
    from urllib.parse import urlparse
    from collections import defaultdict
    import datetime as dt
    from datetime import time,datetime
    from dateutil.relativedelta import relativedelta

    df3 = pd.read_csv('browser_history.csv')

    today = dt.date.today()
    first = today.replace(day = 1)
    lweek = today + relativedelta(days = -7)
    lmonth = first + relativedelta(months = -1)
    lsan = first + relativedelta(months=-6)
    lyear = first + relativedelta(years=-1)

    today = today.strftime("%Y-%m-%d")
    lweek = lweek.strftime("%Y-%m-%d")
    lmonth = lmonth.strftime("%Y-%m-%d")
    lsan = lsan.strftime("%Y-%m-%d")
    lyear = lyear.strftime("%Y-%m-%d")

    print(today)
    print(lweek)
    print(lmonth)
    print(lsan)
    print(lyear)

    c = []

    for i,f,k in zip(df3['Time'],df3['Date'],df3['Url']):
        i1 = datetime.strptime(i, '%H:%M:%S')
        i1 = i1.time()
        if (f >= lyear):
            c.append(f)

    app1 = defaultdict(int)

    for curr in c:
        app1[curr] += 1

    def takeSecond(elem):
        return elem[1]

    def take1(elem):
        return elem[0]

    adk = list(app1.items())
    adk.sort(key = takeSecond, reverse = True)
    df1 = pd.DataFrame(adk,columns = ['Time', 'Count'])
    df1 = df1[df1.Time != '']
    df1.to_csv('lyearcount.csv', sep = ',', index = False)
    print(df1.head(5))
 busyyear()
 def busyyeartop5():
    import matplotlib.pyplot as plt
    import pandas as pd

    import numpy as np

    fig=plt.figure()
    df = pd.read_csv('lyearcount.csv')
    # x-coordinates of left sides of bars  
    left = df['Time'].head(5)
    
    # heights of bars 
    height = df['Count'].head(5)
        

    

    # labels for bars 
    tick_label = df['Time'].head(5)
    plt.xticks(np.arange(1,5), df['Time'].head(5), rotation=30 , fontsize=8)
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.8, color = 'red')


    # naming the x-axis 
    plt.xlabel('URL') 
    # naming the y-axis 
    plt.ylabel('Count') 
    # plot title 
    plt.title('This Year top 5') 
    # function to show the plot 
    fig.savefig("bytop5.png")
 busyyeartop5()

def busyleast():
 def busyweekl5():
    import matplotlib.pyplot as plt
    import pandas as pd

    import numpy as np

    fig=plt.figure()
    df = pd.read_csv('lweekcount.csv')
    # x-coordinates of left sides of bars  
    left = df['Time'].tail(5)
    
    # heights of bars 
    height = df['Count'].tail(5)
        
    # labels for bars 
    tick_label = df['Time'].tail(5)
    plt.xticks(np.arange(1,5), df['Time'].tail(5),rotation=30 , fontsize=8)
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.8, color = 'red')


    # naming the x-axis 
    plt.xlabel('Date') 
    # naming the y-axis 
    plt.ylabel('Count') 
    # plot title 
    plt.title('This Week Least 5 Busy Days') 
    # function to show the plot  
    fig.savefig("lwbusyleast5.png")
 busyweekl5()
 
 def busy6monthl5():

    import matplotlib.pyplot as plt
    import pandas as pd

    import numpy as np

    fig=plt.figure()
    df = pd.read_csv('lsancount.csv')
    # x-coordinates of left sides of bars  
    left = df['Time'].tail(5)
    
    # heights of bars 
    height = df['Count'].tail(5)
        

    

    # labels for bars 
    tick_label = df['Time'].tail(5)
    plt.xticks(np.arange(1,5), df['Time'].tail(5),rotation=30 , fontsize=8)
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.8, color = 'red')


    # naming the x-axis 
    plt.xlabel('Date') 
    # naming the y-axis 
    plt.ylabel('Count') 
    # plot title 
    plt.title('Last six Month busy days least 5') 
    # function to show the plot  
    fig.savefig("lsmleast5.png")
 busy6monthl5()

 def busymonthl5():
   
    import matplotlib.pyplot as plt
    import pandas as pd

    import numpy as np

    fig=plt.figure()
    df = pd.read_csv('lmonthcount.csv')
    # x-coordinates of left sides of bars  
    left = df['Time'].tail(5)
    
    # heights of bars 
    height = df['Count'].tail(5)
        

    

    # labels for bars 
    tick_label = df['Time'].tail(5)
    plt.xticks(np.arange(1,5), df['Time'].tail(5),rotation=30 , fontsize=8)
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.8, color = 'red')


    # naming the x-axis 
    plt.xlabel('Date') 
    # naming the y-axis 
    plt.ylabel('Count') 
    # plot title 
    plt.title('This month Least 5 busy days') 
    # function to show the plot  
    fig.savefig("lmbusyleast5.png")
 busymonthl5()

 def busyyearl5():
        import matplotlib.pyplot as plt
        import pandas as pd

        import numpy as np

        fig=plt.figure()
        df = pd.read_csv('lyearcount.csv')
        # x-coordinates of left sides of bars  
        left = df['Time'].tail(5)
        
        # heights of bars 
        height = df['Count'].tail(5)
            

        

        # labels for bars 
        tick_label = df['Time'].tail(5)
        plt.xticks(np.arange(1,5), df['Time'].tail(5),rotation=30 , fontsize=8)
        # plotting a bar chart 
        plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = 'red')


        # naming the x-axis 
        plt.xlabel('URL') 
        # naming the y-axis 
        plt.ylabel('Count') 
        # plot title 
        plt.title('This Year least 5 busy days') 
        # function to show the plot 
        fig.savefig("lybusyleast5.png")
 busyyearl5()

time()
top5()
least5()
busytop()
busyleast()