import json
import os
import utils

DATA_DIR = "C:\\Users\Sai Teja\Desktop\ELL888-RNN\\CTC"

TEST_DIR = DATA_DIR + "\\test1\\dr"
data = []
with open('document.json') as data_file:    
	data = json.load(data_file)

v = 1
correct_detection_rate =0
false_detection_rate = 0
missed_detection_rate = 0
u =0
deviation = []
while(v<=8):
	#print('$$$$$$$$$$$$$$$$$$')
	#print(v)
	#print('$$$$$$$$$$$$$$$$$$')
	directory = TEST_DIR + str(v)
	for Dir in os.listdir(directory):
		
		#print('+++++++++++++++++++')
		#print(u)
		#print('+++++++++++++++++++')

			
		for p in os.listdir(directory + '\\'+ Dir):
			
			if(p.endswith('.txt')):
				#print(p)
				
				file = utils.read_text_file(directory + '\\'+ Dir+'\\'+p)
				file = utils.normalize_text(file)
				##print(data[file+'_'+str(u)])
				if(len(file)!=0):
					try:

						with open(directory + '\\'+ Dir+'\\'+p[:-4]+'.wrd') as f:
							l = 0
							c_d = 0
							f_d = 0
							m_d = 0
							for line in f:
								if(l!= 0):
									s = line.split()
									j = int(s[0])
									k = int(s[1])
									#print(j,k)
									d = 0
									h = 0
									for i in data[file+'_'+str(u)]:
										if(j<=int(i) and int(i)<=k):
											d = d+1
											h = i
											

									if(d == 1):
										c_d = c_d+1
										deviation = deviation +[h-j]
									elif(d==0):
										m_d = m_d+1
									elif(d>1):
										f_d =f_d+1
								l = l+1
							#print(c_d,m_d,f_d,l-1)
							correct_detection_rate = correct_detection_rate + (c_d/(l-1))
							false_detection_rate = false_detection_rate + (f_d/(l-1))
							missed_detection_rate = missed_detection_rate + (m_d/(l-1))
					except FileNotFoundError: 

						pass
		u = u+1


				
	v=v+1

				
			
print(u)
	
print(correct_detection_rate/1680,missed_detection_rate/1680,false_detection_rate/1680)
#print(deviation)

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

x = deviation

# the histogram of the data
n, bins, patches = plt.hist(x)
plt.xlabel("Deviation from actual boundary")
plt.ylabel("Indices of test data in test files")
plt.show()