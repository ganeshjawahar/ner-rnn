test_info_file='../data/preprocessed/test.tsv'
result_file='../data/preprocessed/result.tsv'
final_file='../data/preprocessed/prediction.tsv'
#Execute  bc-evaluate --INT ../data/preprocessed/prediction.tsv ../data/raw/chemdner_cemp_gold_standard_train_eval.tsv chemdner_cemp_gold_standard_train_eval.tsv > prediction.eval
res_file=open(final_file,'w')
info_lines=[line.rstrip('\n') for line in open(test_info_file)]
pred_lines=[line.rstrip('\n') for line in open(result_file)]
cur_pred=0
for i in xrange(0,len(info_lines),3):
	pId=info_lines[i]
	title=info_lines[i+1].strip().split('\t')
	title_pred=pred_lines[cur_pred].strip().split('\t')
	run=0
	for j in xrange(0,len(title)):
		title_content=title[j].split('$$$')
		pred_content=title_pred[j].strip().split('$$$')
		if pred_content[1]=='2':
			run=run+1
			end_index=str(int(title_content[1])+int(title_content[2]))
			res_file.write(pId+'\tT:'+title_content[1]+':'+end_index+'\t'+str(run)+'\t'+pred_content[0]+'\t'+title_content[0]+'\n')
	abstract=info_lines[i+2].strip().split('\t')
	abstract_pred=pred_lines[cur_pred+1].strip().split('\t')
	for j in xrange(0,len(abstract)):
		abs_content=abstract[j].split('$$$')
		pred_content=abstract_pred[j].strip().split('$$$')
		if pred_content[1]=='2':
			run=run+1
			end_index=str(int(abs_content[1])+int(abs_content[2]))
			res_file.write(pId+'\tA:'+abs_content[1]+':'+end_index+'\t'+str(run)+'\t'+pred_content[0]+'\t'+abs_content[0]+'\n')
	cur_pred+=2
res_file.close()