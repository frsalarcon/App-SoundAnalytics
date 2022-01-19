
import streamlit as st
##########
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
from tensorflow import keras
import matplotlib.pyplot as plt
import soundfile as sf
from keras.models import load_model
import librosa
from librosa import display


#################


EJ=[]
predict=[]
ñ=[]
ñ1=[]  # contador para el grafico empresa
ñ0=[] # exterior
def main(): 
    with st.sidebar:
        st.sidebar.title("SOUNDANALYTICS")
        st.write("App para realizar pruebas usando los audios descargados desde la pagina de SoundAnalytics. \n Version actual del modelo: 0.03.")
    st.title('SoundAnalytics')

    uploaded_file=st.file_uploader(label="Selecciona una muestra para analizar. (Audio en ogg)", type=["ogg"])
    if uploaded_file is not None:
        data, samplerate = sf.read(uploaded_file)

  
        st.write(print(uploaded_file))
        st.write("Audio:")
        st.audio(uploaded_file)
        st.write("Gráfica de la señal:")
        tiempo=np.linspace(0,len(data)/44100, num=len(data))
        plt.figure(figsize=(10,4))
        
        plt.tick_params(labelleft=False)
        plt.tight_layout(pad=0.05)
        plt.plot(tiempo, data)
        plt.xlim(0,tiempo[-1])

        plt.xlabel('Tiempo (s)', fontsize=10)
        plt.ylabel('Amplitud de señal', fontsize=20)
        
        st.pyplot(plt)
        plt.close()





        plt.figure(figsize=(10,4))

        plt.specgram(data, NFFT=512, Fs=44100, scale='dB', cmap='viridis')
        plt.ylim(0,15000)
        plt.xlabel('Tiempo (s)', fontsize=10)
        plt.ylabel('Frecuencias (Hz)', fontsize=10)
        st.pyplot(plt)
        plt.close()
        
        
        
        
        
        
        
        
        
        

        
        
        split_wave=np.array_split(data,len(data)/samplerate)
        for j in range(len(data)//samplerate):
            signals = tf.reshape(split_wave[j], [1, -1])
            spec=tfio.audio.spectrogram(signals, nfft=512, window=512, stride=256)
            EJ.append(spec)

            
        model_relu=load_model('0.045v.h5')

        pp=tf.expand_dims(EJ, -1)
        for i in range(len(pp)):
            predict.append(model_relu.predict(pp[i])[0])

        predict_t=np.transpose(predict)


        for i in range(len(predict)):
            arg=np.argmax(predict[i])
            if arg == 2:
                ñ.append([0,1])
                ñ1.append(1)
            if arg == 5:
                ñ.append([0,1])
                ñ1.append(1)

            if arg == 7:
                ñ.append([0,1])
                ñ1.append(1)


            else:
                ñ.append([1,0])
                ñ0.append(1)

            ññ=ñ
            ññ=np.transpose(ñ)
        st.write("Grafica de predicción")
        plt.figure(figsize=(10,4))
        yticks = range(0,2,1)

        plt.imshow(ññ, aspect='auto', interpolation='nearest', cmap='Greys')
        plt.yticks(yticks, ['Exterior', 'Empresa'],fontsize=11)
        xticks=np.arange(0,len(EJ),10)
        plt.xlabel('Tiempo (s)', fontsize=10)
        plt.show()
        st.pyplot(plt)
        plt.close()


        cl=['E-aves_fondo',
 'E-grillo',
 'I-grillo_auto_camion',
 'E-grillo-motorauto',
 'E-grillo-viento',
 'I-aves_fondo_Indus',
 'E-queltehues',
 'I-troncos',
 'E-fondo']
        cl=cl[::-1]
        st.write("Grafica de predicción")
        plt.figure(figsize=(10,4))
        yticks = range(0,9,1)


        kk=np.transpose(ñ)
        plt.imshow(kk, aspect='auto', interpolation='nearest', cmap='Greys')
        plt.yticks(yticks, cl ,fontsize=11)
        xticks=np.arange(0,len(EJ),10)
        plt.xlabel('Tiempo (s)', fontsize=10)
        plt.show()
        st.pyplot(plt)
        plt.close()


        p1=len(ñ0)
        p2=len(ñ1)

        langs = [ 'Exterior', 'Empresa']
        students = [p1,p2]

        plt.bar(langs,students, color=['red','blue'])
        plt.xlabel('Cantidad', fontsize=15)
        plt.ylabel('Muestras por segundos', fontsize=15)
        plt.show()
        st.pyplot(plt)
        plt.close()


        
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Exterior', 'Empresa'
        sizes = [p2, p1]
        explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.figsize=(20,20)
        ax1.pie(students, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()
        st.pyplot(plt)
        plt.close()
       
        st.write('Segundos de empresa: ', p2)
        st.write('Segundos de exterior: ', p1)
if __name__ == '__main__':
    main()
       
