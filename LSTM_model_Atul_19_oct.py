
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm # Progress bar

# for scaling

# For scaling, feature selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split 


# For LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from keras.models import load_model

# For TCLab
import tclab
from matplotlib.animation import FuncAnimation

import tkinter as tk
from tkinter import Label, Button, Entry
import pickle

TCLab = tclab.setup(connected=False)

host = "192.168.1.102"
path = "/rpc/pot1/read"

new_model = tf.keras.models.load_model('lstm_control.h5')

new_df = pd.read_csv('PID_train_data.csv')


print(new_df)

# Scale data
X = new_df[['Tsp','err']].values
y = new_df[['Q1']].values
s_x = MinMaxScaler()
Xs = s_x.fit_transform(X)

s_y = MinMaxScaler()
ys = s_y.fit_transform(y)
window = 15

'''
# Load model and related data from PKL file
model_info = pickle.load(open('lstm_control_27_oct.pkl', 'rb'))
new_model = tf.keras.models.load_model(model_info[0])
s_x = model_info[1]
s_y = model_info[2]
window = model_info[3]

'''
# Show the model architecture
# print(new_model.summary())
import http.client

#function call tO read T1 from the board

def readcurrenttemperature():
    connection = http.client.HTTPConnection(host)
    connection.request("GET", path)
    response = connection.getresponse()

    x1= response.read().decode()
    x=float(x1)
    x = float("{:.3f}".format(x))
    T1= x*100
    T1 =1+ float("{:.2f}".format(T1*1.030))
     #offset is 1
    print('T1: ',T1)
    
    return(T1)
    connection.close()
  
    
def avg_readcurrenttemperature():
    #ii=0
    #TT=0
    #while (ii<5):
      #ii=ii+1
    TT=readcurrenttemperature()
      
       
      #print("I=",i)
    print("done")
    return(TT)


#function call to write voltage to the board
def writetoDAC0to10volt(input):
  connection = http.client.HTTPConnection(host)
  connection.request("GET", path)
  response = connection.getresponse()
  str1="/rpc/anout/write%"
  input=200+input
  str2=str(input)
  
  path2= str1+str2
  connection.request("GET", path2)
  print(path2)
  x=float(input)
  x=x-200
  print("The voltage written 0 to 5 is =",x*3.3*3)#1.5 changed to 3
  connection.close()

    
# LSTM controller code
def lstm(T1_m, Tsp_m):
    # Calculate error (necessary feature for LSTM input)
    err = Tsp_m - T1_m

    # Format data for LSTM input
    X = np.vstack((Tsp_m,err)).T
    Xs = s_x.transform(X)
    Xs = np.reshape(Xs, (1, Xs.shape[0], Xs.shape[1]))

    # Predict Q for controller and unscale
    Q1c_s = new_model.predict(Xs)
    Q1c = s_y.inverse_transform(Q1c_s)[0][0]

    # Ensure Q1c is between 0 and 100
    Q1c = np.clip(Q1c,0.0,100.0)
    return Q1c





# Define a function to save data to a CSV file
def save_data_to_csv(i, i_values, Tsp_values, T1_values, Qlstm_values):
    data = {
        'i_values': i_values,
        'Tsp_values': Tsp_values,
        'T1_values': T1_values,
        'Qlstm_values': Qlstm_values
    }

    df = pd.DataFrame(data)
    csv_file = f"output_{i // 3600}.csv" #3600
    df.to_csv(csv_file, index=False)
    print(f"Data has been saved to {csv_file}")



# Run time in minutes
run_time = 24*60

# Number of cycles
loops = int(60.0*run_time)

# arrays for storing data
T1 = np.zeros(loops) # measured T (degC)
Qlstm = np.zeros(loops) # Heater values for LSTM controller
tm = np.zeros(loops) # Time

# Temperature set point (degC)
with TCLab() as lab:
    Tsp = np.ones(loops) * lab.T1

# vary temperature setpoint
end = window #+ 15 # leave 1st window + 15 seconds of temp set point as room temp
Tsp[end:]=40


manual_flag = True
manual_val= 10
sp_g = 40

crr_LSTM_Q = 0
crr_tem = 0
crr_tsp = 0

def mann_update():
    itrator= 0
    global manual_flag
    global manual_val
    manual_flag = True
    manual_heating = manual_heat.get()
    manual_val = int(manual_heating)
    manual.configure(bg="green")
    auto.configure(bg="red")

def lstm_auto():
    global manual_flag
    manual_flag = False
    manual.configure(bg="red")
    auto.configure(bg="green")
    
# Function to update Tsp values
def update_Tsp():
    global itrator
    global Tsp
    end = window #+ 15
    start = end
    #end += 1000
    set_point = sp.get()
    sp_g = int(set_point)
    Tsp[itrator+1:] = sp_g #Tsp[start:] = sp_g
    print("Set Point:", set_point)

# Create a Tkinter window
root = tk.Tk()
root.title("Level Setpoint(Tsp) control panel")

label = tk.Label(root, text="Set Point:")
label.grid(row=2, column=0)
sp = tk.Entry(root)
sp.grid(row=2, column=2)
sp.insert(0, sp_g)

# Create a button to update Tsp values
update_button = tk.Button(root, text="Update Tsp", command=update_Tsp)
update_button.grid(row=3, column=0, columnspan=2)

# Label and Entry for Manual Value
manual_label = tk.Label(root, text="Manual Value:")
manual_label.grid(row=4, column=0)
manual_heat = tk.Entry(root)
manual_heat.grid(row=4, column=2)

manual_heat.insert(0, manual_val)

manual = tk.Button(root, text="Manual Override", command=mann_update)
auto = tk.Button(root, text="LSTM auto", command=lstm_auto)

manual.configure(bg="green")
auto.configure(bg="red")

manual.grid(row=5, column=0, columnspan=2)
auto.grid(row=5, column=1, columnspan=2)


# Create an Entry widget for 'y'
crr_lstm_label = tk.Label(root, text=f"y = {crr_LSTM_Q}")
crr_lstm_label.grid(row=7, column=0)

crr_man_label = tk.Label(root, text=f"y = {manual_val}")
crr_man_label.grid(row=8, column=0)

crr_tem_label = tk.Label(root, text=f"y = {crr_tem}")
crr_tem_label.grid(row=9, column=0)

crr_tsp_label = tk.Label(root, text=f"y = {crr_tsp}")
crr_tsp_label.grid(row=10, column=0)



# Create a figure and axes for the real-time plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_ylim((0, 100))
ax.set_xlabel('Time (s)', size=14)
ax.tick_params(axis='both', which='both', labelsize=12)

# Initialize data arrays for real-time plot
i_values = []
Tsp_values = []
T1_values = []
Qlstm_values = []

# Create empty lines for the plot
line_sp, = ax.plot([], 'k-', label='SP (%)') #$(^oC)$
line_t1, = ax.plot([], 'r-', label='$Linac_1$ (%)') #$(^oC)$
line_lstm, = ax.plot([], 'g-', label='$Q_{LSTM}$ (%)')
ax.legend(loc='upper right', fontsize=14)

# Initialize the plot
plt.ion()  # Turn on interactive mode
plt.show()

# Run test
with TCLab() as lab:
    # Find current T1, T2
    print('Temperature 1: {0:0.2f} °C'.format(lab.T1))
    print('Temperature 2: {0:0.2f} °C'.format(lab.T2))

    start_time = 0
    prev_time = 0

    for i, t in enumerate(tclab.clock(loops)):
        itrator= i
        tm[i] = t
        dt = t - prev_time

        # Read temperature (C)
        T1[i] = avg_readcurrenttemperature()#lab.T1 #avg_readcurrenttemperature()#

        # Run LSTM model to get Q1 value for control
        if i >= window:
            # Load data for model
            T1_m = T1[i - window:i]
            Tsp_m = Tsp[i - window:i]
            # Predict and store LSTM value for comparison
            Qlstm[i] = lstm(T1_m, Tsp_m)
            #writetoDAC0to10volt(Qlstm[i]/100)


        crr_LSTM_Q = round(Qlstm[i], 2)
        crr_tem = round(T1[i], 2)
        crr_tsp = round(Tsp[i], 2)

   
        # Write heater output (0-100)
        print("Now flag=:",manual_flag) 

        #setting changing values in GUI
        crr_lstm_label.config(text=f"Current LSTM Q= {crr_LSTM_Q}") 
        crr_man_label.config(text=f"Current Manual Q= {manual_val}") 
        crr_tem_label.config(text=f"Current PV= {crr_tem}") 
        crr_tsp_label.config(text=f"Current Tsp= {crr_tsp}")

        if manual_flag == True:
            writetoDAC0to10volt(manual_val/100)#lab.Q1(manual_val)#writetoDAC0to10volt(manual_val/100)
            Qlstm_values.append(manual_val)#Qlstm_values.append(manual_val)
            print('Q man_voltage: ',(manual_val*10/100))
        else:
            writetoDAC0to10volt(Qlstm[i]/100)#lab.Q1(Qlstm[i])#writetoDAC0to10volt(Qlstm[i]/100)#
            Qlstm_values.append(Qlstm[i])
            print('Q LSTM_voltage: ', (Qlstm[i]*10)/100)

        # Update plot data
        i_values.append(i)
        Tsp_values.append(Tsp[i])
        T1_values.append(T1[i])
        #Qlstm_values.append(Qlstm[i])#additional added

        # Update plot lines
        line_sp.set_data(i_values, Tsp_values)
        line_t1.set_data(i_values, T1_values)
        line_lstm.set_data(i_values, Qlstm_values)

        plt.title(label="IUAC ARTIFICIAL INTELLIGENCE BASED RNN ENGINE", fontsize = 25, color = 'green', loc='center')#Artificial Intelligence Based RNN Engine
        ax.relim()
        ax.autoscale_view()
        fig.canvas.flush_events()  # Update the plot

        prev_time = t

        if i % 3600 == 0 and i != 0: #3600
            # Save data every 3600 loops
            save_data_to_csv(i, i_values, Tsp_values, T1_values, Qlstm_values)

            # Clear the lists to free up memory
            i_values.clear()
            Tsp_values.clear()
            T1_values.clear()
            Qlstm_values.clear()

# Turn off interactive mode after the loop

plt.ioff()
plt.show()
