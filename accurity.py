import matplotlib.pyplot as plt
import numpy as np

# draw the picture
def draw(Curve_one, Curve_two):
    plt.figure()
    plot1, = plt.plot(Curve_one[0], Curve_one[1])
    plot2, = plt.plot(Curve_two[0], Curve_two[1])
    #plot3, = plt.plot(Curve_three[0], Curve_three[1])
    #plot4, = plt.plot(Curve_four[0], Curve_four[1])

    plt.xlabel("step", fontsize="x-large")
    plt.ylabel("loss",    fontsize="x-large")
    
    # set figure information
    plt.title("loss", fontsize="x-large")
    plt.legend([plot1, plot2], ("ac", "normal"), loc="lower right")
    plt.grid(True)
 
    # draw the chart
    plt.show()
    plt.imsave('./0.1结果')
 
def readdata(filename):
    f=open(filename,'r')
    lines = f.readlines()
    data=[]
    for line in lines:
        y=line.split(',')[1]
        data.append(float(y))
    f.close()
    return data

# main function
def main():
    x=[]
    for i in range(0,250):
        x.append(i)
    x1=[]
    for i in range(0,251):
        x1.append(i)
    
    nnfile = './lstm/result/a0.1c1.1.txt'
    nfile = './lstm/result/normal.txt'
    #nfile = './lstm/result/c1.1a0.5.txt'
    #n1file = '/home/bly/文档/结果/0.1/result.txt'
    #n2file = '/home/bly/文档/结果/0.5/result.txt'

    y1 = readdata(nnfile)
    Curve_one=[]
    Curve_one.append(x1)
    Curve_one.append(y1)

     # Curve two
    y2 = readdata(nfile)
    Curve_two  = []
    Curve_two.append(x)
    Curve_two.append(y2)
 
    # Curve three
    '''
    y3=readdata(n1file)
    Curve_three = []
    Curve_three.append(x)
    Curve_three.append(y3)
    
    # Curve four
    y4 = readdata(n2file)
    Curve_four = []
    Curve_four.append(x)
    Curve_four.append(y4[:60])
    '''
    # Call the draw function
    draw(Curve_one, Curve_two)
 
 
# function entrance
if __name__ == "__main__":
    main()
