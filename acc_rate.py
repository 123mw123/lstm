import json
import os
import utils

DATA_DIR = "C:\\Users\Sai Teja\Desktop\ELL888-RNN\\CTC"

TEST_DIR = DATA_DIR + "\\TEST\\DR"
data = []
with open('document.json') as data_file:    
	data = json.load(data_file)
    #print(data.keys())
for key in data.keys():
	if(key.endswith('_0')):
		print(key)
v = 1
correct_detection_rate =0
false_detection_rate = 0
missed_detection_rate = 0
u =0
while(v<=8):
	print('$$$$$$$$$$$$$$$$$$')
	print(v)
	print('$$$$$$$$$$$$$$$$$$')
	directory = TEST_DIR + str(v)
	for Dir in os.listdir(directory):
		
		print('+++++++++++++++++++')
		print(u)
		print('+++++++++++++++++++')

			
		for p in os.listdir(directory + '\\'+ Dir):
			
			if(p.endswith('.TXT')):
				print(p)
				
				file = utils.read_text_file(directory + '\\'+ Dir+'\\'+p)
				file = utils.normalize_text(file)
				print(file)
				print(data[file+'_'+str(u)])
				with open(directory + '\\'+ Dir+'\\'+p[:-4]+'.WRD') as f:
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
							
							for i in data[file+'_'+str(u)]:
								if(j<=int(i) and int(i)<=k):
									d = d+1
									

							if(d == 1):
								c_d = c_d+1
							elif(d==0):
								m_d = m_d+1
							elif(d>1):
								f_d =f_d+1
						l = l+1
					print(c_d,m_d,f_d,l-1)
					correct_detection_rate = correct_detection_rate + (c_d/(l-1))
					false_detection_rate = false_detection_rate + (f_d/(l-1))
					missed_detection_rate = missed_detection_rate + (m_d/(l-1))
		u = u+1


				
	v=v+1

				
			

	
print(correct_detection_rate,missed_detection_rate,false_detection_rate)