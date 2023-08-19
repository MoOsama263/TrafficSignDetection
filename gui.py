import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from gtts import gTTS
import os
#load the trained model to classify sign
from keras.models import load_model
model = load_model('my_model.h5')
#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (5km/h)',
            2:'Speed limit (15km/h)',
            3:'Speed limit (30km/h)',
            4:'Speed limit (40km/h)',
            5:'Speed limit (50km/h)',
            6:'Speed limit (60km/h)',
            7:'End of speed limit (70km/h)',
            8:'End of speed limit (80km/h)',
            9:'Dont Go straight or left',
            10:'Dont Go straight or Right',
            11:'Dont Go straight',
            12:'Dont Go Left',
            13:'Dont Go Left or Right',
            14:'Dont Go Right',
            15:'Dont overtake from Left',
            16:'No Uturn',
            17:'No Car',
            18:'No horn',
            19:'Speed limit (40km/h)',
            20:'Speed limit (50km/h)',
            21:'Go straight or right',
            22:'Go straight',
            23:'Go Left',
            24:'Go Left or right',
            25:'Go Right',
            26:'keep Left',
            27:'keep Right',
            28:'Roundabout mandatory',
            29:'watch out for cars',
            30:'Horn',
            31:'Bicycles crossing',
            32:'Uturn',
            33:'Road Divider',
            34:'Traffic signals',
            35:'Danger Ahead',
            36:'Zebra Crossing',
            37:'Bicycles crossing',
            38:'Children crossing',
            39:'Dangerous curve to the left',
            40:'Steep descent ahead',
            41:'Steep ascent ahead',
            42:'Slow traffic safety slogan',
            43:'Road junction on the right',
            44:'Go right or straight',
            45: 'Go left or straight',
            46: 'electricity warning logo',
            47: 'ZigZag Curve',
            48: 'Train Crossing',
            49: 'Under Construction',
            50: 'fence gate road sign',
            51: 'Fences',
            52: 'Heavy Vehicle Accidents',
            53: 'Unknown6',
            54: 'absolute ban on stopping',
            55: 'No stopping',
            56: 'No entry',
            57: 'Give way',
            58: 'No entry',
            }
#initialise GUI




top=tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#146C94')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
   # pred = model.predict_classes([image])[0]
    pred= numpy.argmax(model.predict([image])[0], axis=-1)
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#011638', text=sign)
    audio = gTTS(text=sign, lang="en", slow=False)
    audio.save("example.mp3")
    os.system("start example.mp3")
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#19A7CE', foreground='white',font=('arial',20,'bold'))
    classify_b.place(relx=0.6,rely=0.8)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/7.25),(top.winfo_height()/7.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload an image",command=upload_image,padx=15,pady=10)

upload.configure(background='#19A7CE', foreground='#FEFAE0',font=('arial',17,'bold'))
upload.place(relx=0.1,rely=0.8)
sign_image.place(relx=0.45,rely=0.3)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Welcome to Traffic Sign Detection System",pady=20, font=('arial',20,'bold'))
heading.configure(background='#19A7CE',foreground='#FEFAE0')
heading.pack()
top.mainloop()