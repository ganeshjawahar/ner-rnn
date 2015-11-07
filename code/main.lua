--[[

Main class for executing RNN and LSTM on CEMP dataset

]]--

require 'torch'
require 'io'
require 'nn'
require 'rnn'
require 'os'
require 'xlua'
require 'optim'

local utils=require 'utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Identify chemical named entity mentions using RNN or GRU or LSTM system')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-train_file','train.tsv','training set file location')
cmd:option('-dev_file','dev.tsv','dev set file location')
cmd:option('-test_file','test.tsv','test set file location')
-- model params (general)
cmd:option('-model_type','rnn','which recurrent model to use? rnn or gru or lstm')
cmd:option('-wdim',300,'dimensionality of word embeddings')
cmd:option('-min_freq',5,'words that occur less than <int> times will not be taken for training')
-- optimization
cmd:option('-learning_rate',0.01,'learning rate')
cmd:option('-max_epochs',100,'number of full passes through the training data')
cmd:option('-reg',1e-4,'regularization parameter l2-norm')
cmd:option('-clip',5,'threshold to clip gradients')
cmd:option('-pre_train',0,'initalize word embeddings with pre-trained word vectors')
-- GPU/CPU
cmd:option('-gpu',1,'1=use gpu; 0=use cpu;')
-- Book-keeping
cmd:option('-print_params',0,'output the parameters in the console. 0=dont print; 1=print;')

-- parse input params
params=cmd:parse(arg)

if params.print_params==1 then
	-- output the parameters	
	for param, value in pairs(params) do
	    print(param ..' : '.. tostring(value))
	end
end
if params.gpu==1 then
	require 'cutorch'
	require 'cunn'
	require 'cunnx'
end

params.vocab={} -- word frequency map
params.index2word={}
params.word2index={}

-- build the vocabulary
utils.buildVocab(params)

-- load data to main memory
utils.loadDataTensors(params) 

-- define the model
params.model=nn.Sequential()
params.word_vecs=nn.LookupTable(#params.index2word,params.wdim)
if params.model_type=='rnn' then
	params.rnn=nn.Recurrent(params.wdim,params.word_vecs,nn.Linear(params.wdim,params.wdim),nn.Sigmoid())
	params.model:add(nn.Sequencer(params.rnn))
elseif params.model_type=='lstm' then
	params.rnn=nn.FastLSTM(params.wdim,params.wdim)
	params.model:add(nn.Sequencer(params.word_vecs))
	params.model:add(nn.Sequencer(params.rnn))
end
params.model:add(nn.Sequencer(nn.Linear(params.wdim,2)))
params.model:add(nn.Sequencer(nn.LogSoftMax()))
params.criterion=nn.SequencerCriterion(nn.ClassNLLCriterion(torch.Tensor{0.03,0.97}))
params.soft=nn.SoftMax()
if params.gpu==1 then
	params.model=params.model:cuda()
	params.criterion=params.criterion:cuda()
	params.soft=params.soft:cuda()
end
if params.pre_train==1 then
	utils.initWordWeights(params,'/home/ganesh/Downloads/t2v/vectors.840B.300d.txt')
end

-- train the model
local idx=torch.randperm(#params.train_input_tensors)
print('Training ...')
local start=sys.clock()
params.best_dev_model=params.model
params.best_dev_score=-1.0
par,grad_params=params.model:getParameters()
for epoch=1,params.max_epochs do
	print('Epoch '..epoch..' ...')
	local epoch_start=sys.clock()
	local epoch_loss=0
	local iteration=0
	xlua.progress(1,#params.train_input_tensors)
	for i=1,#params.train_input_tensors do
		if i%20==0 then
			xlua.progress(i,#params.train_input_tensors)
		end
		local output=params.model:forward({params.train_input_tensors[i]})
		local err=params.criterion:forward(output,{params.train_target_tensors[i]})
		epoch_loss=epoch_loss+err
		iteration=iteration+1
		local gradOutput=params.criterion:backward(output,{params.train_target_tensors[i]})
		params.model:backward({params.train_input_tensors[i]},gradOutput)
		params.model:backwardThroughTime()
		params.model:updateParameters(params.learning_rate)
		params.model:zeroGradParameters()
		if grad_params:norm()>params.clip then
			grad_params:mul(params.clip/grad_params:norm())
		end
	end
	xlua.progress(#params.train_input_tensors,#params.train_input_tensors)
	-- Compute dev. score
	print('Computing dev score ...')
	local tp,tn,fp,fn=0,0,0,0
	for i=1,#params.dev_input_tensors do
		xlua.progress(i,#params.dev_input_tensors)
		local input_tensor={params.dev_input_tensors[i]}
		local target_tensor={params.dev_target_tensors[i]}
		local output=params.model:forward(input_tensor)
		local out=params.soft:forward(output[1])
		local tar=target_tensor[1]
		for j=1,(#out)[1] do
			local pred=1
			if out[j][1]<out[j][2] then
				pred=2
			end
			if pred==1 and tar[j]==1 then
				tn=tn+1
			elseif pred==1 and tar[j]==2 then
				fn=fn+1
			elseif pred==2 and tar[j]==1 then
				fp=fp+1
			else
				tp=tp+1
			end			
		end
	end
	xlua.progress(#params.dev_input_tensors,#params.dev_input_tensors)
	local precision,recall=(tp/(tp+fp)),(tp/(tp+fn))
	local fscore=((2*precision*recall)/(precision+recall))
	print(string.format("Epoch %d done in %.2f minutes. loss=%f Dev Score=(P=%.2f R=%.2f F=%.2f)\n",epoch,((sys.clock()-epoch_start)/60),(epoch_loss/iteration),precision,recall,fscore))
	if fscore>params.best_dev_score then
		params.best_dev_score=fscore
		params.best_dev_model=params.model:clone()
	end
end
print(string.format("Done in %.2f minutes.",((sys.clock()-start)/60)))

-- Do the final testing
print('Computing test score ...')
local tp,tn,fp,fn=0,0,0,0
local start=sys.clock()
for i=1,#params.test_input_tensors do
	xlua.progress(i,#params.test_input_tensors)
	local input_tensor={params.test_input_tensors[i]}
	local output=params.best_dev_model:forward(input_tensor)
	local out=params.soft:forward(output[1])
	local tar=params.test_target_tensors[i]
	for j=1,(#out)[1] do
		local pred=1
		if out[j][1]<out[j][2] then
			pred=2
		end
		if pred==1 and tar[j]==1 then
			tn=tn+1
		elseif pred==1 and tar[j]==2 then
			fn=fn+1
		elseif pred==2 and tar[j]==1 then
			fp=fp+1
		else
			tp=tp+1
		end		
	end
end
xlua.progress(#params.test_input_tensors,#params.test_input_tensors)
local precision,recall=(tp/(tp+fp)),(tp/(tp+fn))
local fscore=((2*precision*recall)/(precision+recall))
print(string.format('Test Score=(P=%.2f R=%.2f F=%.2f)',precision,recall,fscore))
print(string.format("Testing Done in %.2f minutes.",((sys.clock()-start)/60)))
