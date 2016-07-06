FILE_NAME = raw_input("enter filename : ")
MAX_ALLOWED_SIZE = 50

def findSize():

	d = dict()
	temp = 0
	count = 0
	for line in open(FILE_NAME):
		if "-DOCSTART-" in line:
			count += 1
			temp = 0
			continue

		if line in ['\n', '\r\n']:
			if temp == 0:
				continue
			if temp in d:
				d[temp] += 1	
			else:
				d[temp] = 1
			temp = 0
		else:
			temp += 1
	print count

	print d
	temp = 1
	for i in d:
		if i < 50:
			temp += d[i]

	print temp

#findSize()

def noOfTag():
	d = dict()
	count = 0
	for line in open(FILE_NAME):
		if len(line.split()) == 4:
			if line.split()[3] in d:
				d[line.split()[3]] += 1
			else:
				d[line.split()[3]] = 1

			count += 1		
	print "count = " + str(count)
	print d


#noOfTag()

def removeCrap():

	f = open(FILE_NAME)
	a = f.readlines()
	l = list()

	for i in range(len(a)):
		if "-DOCSTART-" in a[i]:
			pass
		else:
			l.append(a[i])

	ff = open('nocraptestf','w')
	ff.writelines(l)

removeCrap()


def modifyDataSize():

	final_list = list()
	l = list()
	temp = 0	
	count = 0
	for line in open('nocraptestf', 'r'):
		if line in ['\n', '\r\n']:
			if temp == 0:
				l = []
			elif temp > MAX_ALLOWED_SIZE:
				count += 1
				l = []
				temp = 0
			else:
				l.append(line)
				final_list.append(l)
				l = []
				temp = 0
		else:
			l.append(line)
			temp += 1

	f = open("50testb",'w')
	print len(final_list)
	for i in final_list:
		f.writelines(i)
	print final_list[0:5]
modifyDataSize()


def firstFewSentences():

	NO_SENT_NEEDED = 1000
	l = []
	final_list = []
	temp = 0
	for line in open(FILE_NAME):
		if line in ['\n', '\r\n']:
			if temp < NO_SENT_NEEDED:
				l.append(line)
				final_list.append(l)
				temp += 1
			l = []
		else:
			l.append(line)

	f = open("50trainsmall",'w')
	for i in final_list:
		f.writelines(i)




#firstFewSentences()
