import random
train_lines=[line.rstrip('\n') for line in open('../data/preprocessed/train.tsv')]
dev_lines=[line.rstrip('\n') for line in open('../data/preprocessed/dev.tsv')]
key_list=[]
data_map={}
for i in xrange(0,len(train_lines)/3):
	key_list.append(train_lines[3*i])
	data_map[train_lines[3*i]]=[train_lines[(3*i)+1],train_lines[(3*i)+2]]
for i in xrange(0,len(dev_lines)/3):
	key_list.append(dev_lines[3*i])
	data_map[dev_lines[3*i]]=[dev_lines[(3*i)+1],dev_lines[(3*i)+2]]
total=len(key_list)
train=int((0.7)*total)
dev=int((0.1)*total)
test=total-train-dev
print(total)
print(train)
print(dev)
print(test)
random.shuffle(key_list)
new_train=open('train.tsv','w')
new_dev=open('dev.tsv','w')
new_test=open('test.tsv','w')
for i in xrange(0,len(key_list)):
	if i<train:
		new_train.write(key_list[i]+'\n')
		new_train.write(data_map[key_list[i]][0]+'\n')
		new_train.write(data_map[key_list[i]][1]+'\n')
	elif i<train+dev:
		new_dev.write(key_list[i]+'\n')
		new_dev.write(data_map[key_list[i]][0]+'\n')
		new_dev.write(data_map[key_list[i]][1]+'\n')
	else:
		new_test.write(key_list[i]+'\n')
		new_test.write(data_map[key_list[i]][0]+'\n')
		new_test.write(data_map[key_list[i]][1]+'\n')
new_train.close()
new_dev.close()
new_test.close()