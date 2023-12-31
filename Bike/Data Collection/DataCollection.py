import datetime
import serial
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("tkAgg")

ser = serial.Serial('COM6', 9600)
ser.flushInput()

plot_window = 20
sound_var = np.array(np.zeros([plot_window]))
vibration_var = np.array(np.zeros([plot_window]))
temp_var = np.array(np.zeros([plot_window]))
humid_var = np.array(np.zeros([plot_window]))

plt.ion()
fig, ax = plt.subplots()
ax.set_title('Failure Detection Sound Sensor Project')
ax.set_xlabel('Time')
ax.set_ylabel('Parameters')

soundline, = ax.plot(sound_var)
vibrationline, = ax.plot(vibration_var)
templine, = ax.plot(temp_var)
humidline, = ax.plot(humid_var)
fields = ['Time','Sound','Vibration','Temperature','Humidity']
f = open(".\TrainingData.csv", "a+")
writer = csv.writer(f, delimiter=',')
writer.writerow(fields)
while True:
    try:
        s = ser.readline().decode()
        if s != "":
            rows = [float(x) for x in s.split(',')]
            print(rows)
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rows.insert(0, time)
            print(rows)
            writer.writerow(rows)

            sound_var = np.append(sound_var, rows[1])
            sound_var = sound_var[1:plot_window + 1]
            soundline.set_ydata(sound_var)

            vibration_var = np.append(vibration_var, rows[2])
            vibration_var = vibration_var[1:plot_window + 1]
            vibrationline.set_ydata(vibration_var)

            temp_var = np.append(temp_var, rows[3])
            temp_var = temp_var[1:plot_window + 1]
            templine.set_ydata(temp_var)

            humid_var = np.append(humid_var, rows[4])
            humid_var = humid_var[1:plot_window + 1]
            humidline.set_ydata(humid_var)

            ax.relim()
            ax.autoscale_view()
            ax.legend([soundline, vibrationline, templine, humidline], ['sound', 'vibration', 'temperature', 'humidity'])
            fig.canvas.draw()
            fig.canvas.flush_events()

            f.flush()
    except:
        print("Keyboard Interrupt")
        break