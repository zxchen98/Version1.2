from tkinter import *
import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

from datetime import datetime
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Building the RNN
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM, CuDNNLSTM
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
#from opts import parse_opts
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

from DataRequest import *
from ModelTrain import *

# Time Period
time_steps = 30
temp=None

#Create a window
root = tk.Tk()

####### App Name #######
root.title('AI Stock Prediction For S&P500 LIVE ver 1.1')

# Size of the window
root.geometry('1200x700')

####### Set up Framework #######
# Split the window into two parts
topFrame = Frame(root)
topFrame.pack(side=TOP, padx=100)
bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM, padx=10, pady=10)

topLeftFrame = Frame(root)
topLeftFrame.pack(side=LEFT, padx=150)
topRightFrame = Frame(root)
topRightFrame.pack(side=LEFT)

'''Top Frame Set'''
####### Place Title #######
title = tk.Label(topLeftFrame, text="        Stock Search         ")
title.config(font=("Courier", 20))
title.pack(side=TOP)

InputSearchEntry=tk.Entry(topLeftFrame)
InputSearchEntry.pack(side=TOP)

####### Place Logo #######
path = "./visionx-logo.png"
logo = Image.open(path)
logo = logo.resize((200, 200), Image.ANTIALIAS)
#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
logo = ImageTk.PhotoImage(logo)
#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
panel = tk.Label(topRightFrame, image = logo)
panel.pack(side=RIGHT, pady=15)

''' Bottom Frame Set '''
####### Bottom Frame Split #######
# Split Bot Frame to two parts, for text&chart
BotLeftFrame = Frame(bottomFrame)
BotLeftFrame.pack(side=LEFT, padx=5)
BotMidFrame = Frame(bottomFrame)
BotMidFrame.pack(side=LEFT)
BotRightFrame = Frame(bottomFrame)
BotRightFrame.pack(side=LEFT)

test_close_plot, y_pred_group, y_pred_v, t='', '', '', ''

def search_button():
	global modelfile, MT, dh, df, Stock_symbol
	Stock_symbol = str(InputSearchEntry.get())

	try:
		dh = DataHist(Stock_symbol)
		df = dh.RequestFinaldf()

		print("plotting...")
		####### Draw Chart #######
		f = Figure(figsize=(9, 6), dpi=50)
		a = f.add_subplot(111)
		a.plot(df['date'][-90:,], df['Close'][-90:,], color='red', label='Real intel Stock Price')
		#a.plot(df['date'], df['Close'], color='red', label='Real intel Stock Price')
		a.set_title(str('90 Days of '+ Stock_symbol))
		a.set_xlabel('Date')
		a.set_ylabel(str(Stock_symbol+' Close Price'))

		canvas = FigureCanvasTkAgg(f, BotLeftFrame)
		canvas.draw()
		canvas.get_tk_widget().grid(row=0)
		
		def ShowInfo():
			global InputDate, Pred, Act, Acc, InputDateEntry, pred, act, acc
			####### Bottom Mid Frame for Text #######
			InputDate = tk.Label(BotMidFrame,text='Input Date', fg='black')
			InputDate.grid(row = 0)
			Pred = tk.Label(BotMidFrame,text='', fg='black')
			Pred.grid(row = 1, column=0)
			Act = tk.Label(BotMidFrame,text='', fg='black')
			Act.grid(row = 2, column=0)
			Acc = tk.Label(BotMidFrame,text='', fg='black')
			Acc.grid(row=3, column=0)

			#######Bottom Left Frame for Input/Output #######
			InputDateEntry=tk.Entry(BotMidFrame)
			InputDateEntry.grid(row=0,column=1)
			pred = tk.Label(BotMidFrame, text='', fg='black')
			pred.grid(row=1, column=1)
			act = tk.Label(BotMidFrame, text='', fg='black')
			act.grid(row=2, column=1)
			acc = tk.Label(BotMidFrame, text='', fg='red')
			acc.grid(row=3, column=1)

		ShowInfo()



		####### train Button #######
		def train_button():
			global modelfile, MT
			MT = ModelTrain(df)
			modelfile = MT.GetModel()
			#MT.ModelTest(modelfile)
		'''
		Train_button = tk.Button(BotLeftFrame, text ='Train a Model',font = 'Arial', bg = 'black',activebackground = 'white', width=15, height = 2, command=train_button)
		Train_button.grid(row=1, column=0)
		'''

		####### Load model #######
		def LoadModel():
			global model, MT

			try:
				#load original model
				MT = ModelTrain(df)
				modelfile = MT.GetOriModel()
			except ValueError as detail:
				tk.messagebox.showerror("No Model Warning","There is no model in the folder, please train a model first!")
			
			filepath = str(modelfile)
			model = load_model(filepath)

		# This funcion is use for calculate Accuracy and plot
		def pre_plot():
			global test_close_plot, y_pred_group, y_pred_v, t
			test_date_index = np.where(df[date].values == temp)[0][0]
			test_date = np.array(X_t[test_date_index-30])
			test_date_group = np.array(df[date].values[test_date_index-30:test_date_index])
			t = np.array([datetime.strptime(str(tt).split('T')[0],'%Y-%m-%d')for tt in test_date_group])
			test_info = df[test_date_index-30:test_date_index]
			test_close_plot = test_info[cp]	

			####### Predict that day #######
			y_pred = model.predict([[test_date]])
			y_pred_v = y_sc.inverse_transform(y_pred[0].reshape(-1, 1))[0][0]

			####### Predict pervious 30 days #######
			num_pred_group = len(y) - time_steps
			#shape 30*30*18
			test_pred_group = X_t[num_pred_group-30:num_pred_group]
			y_pred_group = model.predict(test_pred_group)
			y_pred_group = y_sc.inverse_transform(y_pred_group.reshape(-1, 1))



		def Pred_button():
			global temp, X, y, X_sc, y_sc, X_t, y_t, percent

			LoadModel()
			X, y = MT.X, MT.y
			X_sc, y_sc, X_t, y_t = MinMax(X, y)

			temp = np.datetime64(InputDateEntry.get())

			test_date_list = df[date].values
			print(test_date_list[-5:])
			if temp not in test_date_list:
				tk.messagebox.showerror("Error","Not a valid date")
			else:
				test_date_index = np.where(test_date_list == temp)[0][0]
				test_date = np.array(X_t[test_date_index-30])
				test_date_value = np.array(y_t[test_date_index-30])

				test_pred = model.predict([[test_date]])
				test_pred_value = y_sc.inverse_transform(test_pred[0].reshape(-1, 1))[0][0]

				test_real = np.array(y[test_date_index])

				if(test_pred_value>test_real):
					t = 'Increasing'
					#v = (test_pred_value - test_real)/test_real*100
					v = (test_pred_value - test_real)
				elif(test_pred_value < test_real):
					t = 'Decreasing'
					#v = (test_real - test_pred_value)/test_real*100
					v = (test_pred_value - test_real)

				pre_plot()

				Pred.configure(text="Predicted Price")
				Pred.bind("<Enter>", lambda e: e.widget.config(text="2nd day's Price"))
				Pred.bind("<Leave>", lambda e: e.widget.config(text="Predicted Price"))
				Act.configure(text=t)
				Acc.configure(text='')

				pred.configure(text=str(test_pred_value))
				#act.configure(text=str(str(round(v, 2))+"%"))
				act.configure(text=str(round(v, 2)))
				acc.configure(text=str(''))

				Plot_button = tk.Button(BotMidFrame, text ='Plot Chart',font = 'Arial', relief='groove', bg = 'white', command=plot_button)
				Plot_button.grid(row=4,column=1)
				
		'''
		Pred_button = tk.Button(BotMidFrame, text ='Predict',font = 'Arial', bg = 'white',activebackground = 'black', width=10, height = 2, command=Pred_button)
		Pred_button.grid(row=4, column=0, pady=5)
		'''

		def Proof_button():
			global temp, X, y, X_sc, y_sc, X_t, y_t
			LoadModel()
			X, y = MT.X, MT.y
			X_sc, y_sc, X_t, y_t = MinMax(X, y)

			temp = np.datetime64(InputDateEntry.get())

			test_date_list = df[date].values
			print(test_date_list[-5:])
                        
			if temp not in test_date_list:
				tk.messagebox.showerror("Error","Not a valid date")
			else:
				test_date_index = np.where(test_date_list == temp)[0][0]
				#tomorrow's pred value
				test_date = np.array(X_t[test_date_index-30])
				test_date_value = np.array(y_t[test_date_index-30])

				test_pred = model.predict([[test_date]])
				test_pred_value = y_sc.inverse_transform(test_pred[0].reshape(-1, 1))[0][0]

				#today's pred value
				test_today_date = np.array(X_t[test_date_index-31])
				test_today_date_value = np.array(y_t[test_date_index-30])

				test_today_pred = model.predict([[test_today_date]])
				test_today_pred_value = y_sc.inverse_transform(test_today_pred[0].reshape(-1, 1))[0][0]

				#today's real value
				test_real = np.array(y[test_date_index])

			pre_plot()

			Pred.configure(text="Predicted Closing Price")
			Pred.bind("<Enter>", lambda e: e.widget.config(text="Today's  Closing  Price"))
			Pred.bind("<Leave>", lambda e: e.widget.config(text="Predicted Closing Price"))
			Act.configure(text="Actual Closing Price")
			Act.bind("<Enter>", lambda e: e.widget.config(text="Today's Actual Price"))
			Act.bind("<Leave>", lambda e: e.widget.config(text="Actual Closing Price"))
			Acc.configure(text='Accuracy')
			
			pred.configure(text=str(test_today_pred_value))
			act.configure(text=str(test_real))

			####### Accuracy Rate #######
			a=np.array(test_close_plot).reshape(30, 1)
			b=np.array(y_pred_group)
			Accuracy = 100-np.sum(abs(a-b)/a)/time_steps*100
			acc.configure(text=(str(round(Accuracy, 4))+"%"))
		'''
		Proof_button = tk.Button(BotMidFrame, text ='Proof',font = 'Arial', bg = 'white',activebackground = 'black', width=10, height = 2, command=Proof_button)
		Proof_button.grid(row=5, pady = 5)
		'''
		Train_button = tk.Button(BotLeftFrame, text ='Train a Model',font = 'Arial', relief='groove', bg = 'white', command=train_button)
		Train_button.grid(row=1, column=0)
		Pred_button = tk.Button(BotMidFrame, text ='Predict',font = 'Arial', bg = 'white',activebackground = 'black', command=Pred_button)
		Pred_button.grid(row=4, column=0, pady=5)
		Proof_button = tk.Button(BotMidFrame, text ='Proof',font = 'Arial', bg = 'white',activebackground = 'black', command=Proof_button)
		Proof_button.grid(row=5, pady = 5)

	except KeyError:
		tk.messagebox.showerror("Error","Symbol Not Found")
	except ValueError as detail:
		tk.messagebox.showerror("Error",detail)

#Search_button = tk.Button(topLeftFrame, text ='Search',font = ('Arial', 20), bg = 'black',activebackground = 'white', command=search_button)
Search_button = tk.Button(topLeftFrame, text ='Search',font = ('Arial', 20), relief='groove', bg = 'white', command=search_button)

Search_button.pack(side=BOTTOM)


####### Give X, y, get X_sc, y_sc, X_t, y_t ########
def MinMax(X, y):
	X_sc = MinMaxScaler(feature_range=(0,1))
	y_sc = MinMaxScaler(feature_range=(0,1))
	X_MinMax = X_sc.fit_transform(X)
	y_MinMax = y_sc.fit_transform(np.array(y).reshape(-1, 1))

	X_t = []
	y_t = []

	for i in range(time_steps, len(y)):
		X_t.append(X_MinMax[i-time_steps:i])
		y_t.append(y_MinMax[i])
	X_t, y_t = np.array(X_t), np.array(y_t)
	return X_sc, y_sc, X_t, y_t

# default data and cp 
date = 'date'
cp = 'Close'

####### Plot Button #######
def plot_button():
	global percent, InputShadow

	shadow = tk.Label(BotRightFrame,text='Adjust Percent(3% original)', fg='black', bg = 'white')
	shadow.grid(row = 0)

	InputShadow=tk.Entry(BotRightFrame)
	InputShadow.grid(row = 1)

	percent = 3

	####### Draw Chart #######
	percent = int(percent)
	flat_list = [item for sublist in y_pred_group for item in sublist]
	f1 = [i*(1-int(percent)/100) for i in flat_list]
	f2 = [i*(1+int(percent)/100) for i in flat_list]


	f = Figure(figsize=(7, 4), dpi=50)
	a = f.add_subplot(111)
	b = f.add_subplot(111)
	ax = f.add_subplot(111)
	ax.fill_between(t, f1, f2, facecolor='yellow', interpolate=True)
	#ax.fill_between(t, test_close_plot, flat_list, where=flat_list >= test_close_plot, facecolor='yellow', interpolate=True)
	#ax.fill_between(t, test_close_plot, flat_list, where=flat_list <= test_close_plot, facecolor='yellow', interpolate=True)
	a.plot(t, test_close_plot, color='red', label='Real intel Stock Price')
	#b.plot(t, y_pred_group,color='blue', label='Predicted Intel Stock Price')
	a.plot(np.array(datetime.strptime(str(temp), '%Y-%m-%d')), np.array(y_pred_v),marker='x', color='blue', label='Predict Close Price')
	
	a.set_title('intel Stock Price Prediction')
	a.set_xlabel('Time')
	a.set_ylabel('Intel Stock Price')
	a.legend(bbox_to_anchor=(0, 1.02, 1, .102), loc=3,
             ncol=2, borderaxespad=0)

	canvas = FigureCanvasTkAgg(f, BotRightFrame)
	canvas.draw()
	canvas.get_tk_widget().grid(row=2)

	Replot_button = tk.Button(BotMidFrame, text ='Re-Plot',font = 'Arial', relief='groove', bg = 'white', command=replot_button)
	Replot_button.grid(row=5,column=1)

def replot_button():
	global percent
	percent = InputShadow.get()
	
	flat_list = [item for sublist in y_pred_group for item in sublist]
	f1 = [i*(1-int(percent)/100) for i in flat_list]
	f2 = [i*(1+int(percent)/100) for i in flat_list]
	
	f = Figure(figsize=(7, 4), dpi=50)
	a = f.add_subplot(111)
	b = f.add_subplot(111)
	ax = f.add_subplot(111)
	ax.fill_between(t, f1, f2, facecolor='yellow', interpolate=True)
	#ax.fill_between(t, test_close_plot, flat_list, where=flat_list >= test_close_plot, facecolor='yellow', interpolate=True)
	#ax.fill_between(t, test_close_plot, flat_list, where=flat_list <= test_close_plot, facecolor='yellow', interpolate=True)
	a.plot(t, test_close_plot, color='red', label='Real intel Stock Price')
	#b.plot(t, y_pred_group,color='blue', label='Predicted Intel Stock Price')
	a.plot(np.array(datetime.strptime(str(temp), '%Y-%m-%d')), np.array(y_pred_v),marker='x', color='blue', label='Predict Close Price')
	
	a.set_title('intel Stock Price Prediction')
	a.set_xlabel('Time')
	a.set_ylabel('Intel Stock Price')
	a.legend(bbox_to_anchor=(0, 1.02, 1, .102), loc=3,
             ncol=2, borderaxespad=0)

	canvas = FigureCanvasTkAgg(f, BotRightFrame)
	canvas.draw()
	canvas.get_tk_widget().grid(row=2)



root.mainloop()




