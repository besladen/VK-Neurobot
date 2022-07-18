from PIL import Image, ImageDraw, ImageFont

#font="/System/Library/Fonts/Supplemental/Impact.ttf"  #MacOS

def lineSeparation(numoflines:int):
    sepProportion=0.05 #Line separation as a reason of the height of text
    ":parameter numoflines. Total number of text lines"
    ":return 0. The separation within the lines in the upper side"
    ":return 1. The separation within the lines in the down side"
    ":return 2. The total number of separations within lines of text"
    separations=1
    if numoflines==1:
        uplineSeparation=0 #zero lines up
        downlineSeparation=0 #one line up
        separations=0
    elif numoflines==2:
        uplineSeparation=0 #zero lines up
        downlineSeparation=(1/2)*sepProportion #2 lines down
    elif numoflines==3:
        uplineSeparation=0 #one line up
        downlineSeparation=(1/2)*sepProportion #2 lines down
    else:
        uplineSeparation=1/numoflines*sepProportion
        downlineSeparation=1/numoflines*sepProportion
        separations=numoflines-2
    return(uplineSeparation*HtextBox,downlineSeparation*HtextBox,separations)
        

def getText(text:str, fontpath):
    ":parameter text. The original text"
    ":return lines. A list with each line of text"
    ":return nlines. The number of lines"
    ":return font. The PIL Imagefont object"
    words=text.split(" ")
    print(words)
    nlines=0
    downSep, numofSeps=lineSeparation(nlines+1)[1:3]
    maxiters=10 #How many iterations to try before adding another line?


    for _ in range(7): # 7 lines of text max
        nlines+=1 
        MaxSize1=(HtextBox-numofSeps*downSep)/nlines
        downSep2, numofSeps2=lineSeparation(nlines+1)[1:3]
        MaxSize2=(HtextBox-numofSeps2*downSep2)/(nlines+1)
        MinSize1=MaxSize2/1.3 #Arbitrary number, you can test it!

        step=(MaxSize1-MinSize1)/maxiters  #The size step in reducing the font per i iteration
        IfontSize=MaxSize1
        font=ImageFont.truetype(fontpath, int(round(IfontSize,0))) 
        for i in range(maxiters):
            i+=1
            lines=list()
            start=0
            for j in range(len(words)):
                line=" ".join(words[start:j+1])
                Wline=font.getsize(line)[0]
                if Wline>WtextBox: #The line already exceded the textBox capacity
                    start=j
                    line=line[0:-len(words[j])]
                    lines.append(line)
                if nlines<=len(lines): #If whe have more lines in a list than what was stablished, then
                    fontSize=IfontSize-step*i #Try a smaller FontSize
                    font=ImageFont.truetype(fontpath, int(round(fontSize,0))) 
                    break
                if j+1==len(words): #Great! The lines fit without exceding nlines and Wtextbox. 
                    lines.append(line)
                    return(lines, nlines, font)



def memefy(file:str, text:str, fontpath):
    ":parameter file. I.E 'Myimage.jpg'"
    ":parameter text. The original text"
    ":return pic. The PIL image"
    pic=Image.open(file) 

    global Wpic, Hpic, WMargin, HtextBox, WtextBox, HMargin #Variables used in multiple functions
    #-----------------------------------------------------------------------------------#
    #Note on notations. W:Width, H: Heigh
    #Here you can define the text parameters in relation to the proportions of the image. 
    Wpic, Hpic = pic.size
    WMargin=1/15*Wpic 
    HtextBox=Hpic/4 
    WtextBox=Wpic-2*WMargin 
    HMargin=2/15*HtextBox
    #-----------------------------------------------------------------------------------#

    draw=ImageDraw.Draw(pic)

    lines, nlines, font=getText(text, fontpath)
    uplinesep, downlinesep=lineSeparation(nlines)[0:2]

    if nlines>2: 
        uplines=lines[0:int(nlines/2)]
        downlines=lines[int(nlines/2):]
        y=HMargin
        for line in uplines:
            (Wline, Hline)=font.getsize(line)
            dy=uplinesep+Hline
            x=(Wpic-Wline)/2
            draw.text((x,y), line, (255, 255, 255), font)
            y+=dy
    else:
        downlines=lines
    
    Hdownbox=0
    dy=list()
    x=list()
    for line in downlines:
        Wline, Hline=font.getsize(line)
        Hdownbox+=Hline
        dy.append(downlinesep+Hline)
        x.append((Wpic-Wline)/2)
        

    y=Hpic-Hdownbox-HMargin
    for i in range(len(downlines)):
        draw.text((x[i],y), downlines[i], (255, 255, 255), font)
        y+=dy[i]

    return(pic)




