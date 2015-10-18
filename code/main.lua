--[[

Main class for executing RNN and LSTM on CEMP dataset

]]--

require 'torch'
require 'io'
require 'nn'
require 'rnn'
require 'os'
require 'xlua'

local utils=require 'utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Identify chemical named entity mentions using RNN or GRU or LSTM system')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-train_file','../data/preprocessed/train.tsv','training set file location')
cmd:option('-dev_file','../data/preprocessed/dev.tsv','dev set file location')
cmd:option('-test_file','../data/preprocessed/test.tsv','test set file location')
cmd:option('-res_file','../data/preprocessed/result.tsv','result file location')
-- model params (general)
cmd:option('-model_type','rnn','which recurrent model to use? rnn or gru or lstm')
cmd:option('-wdim',50,'dimensionality of word embeddings')
cmd:option('-min_freq',5,'words that occur less than <int> times will not be taken for training')
-- optimization
cmd:option('-learning_rate',0.01,'learning rate')
cmd:option('-batch_size',5,'number of sequences to train on in parallel')
cmd:option('-max_epochs',25,'number of full passes through the training data')
cmd:option('-dropout',1, 'apply dropout after each recurrent layer')
cmd:option('-dropout_prob', 0.5, 'probability of zeroing a neuron (dropout probability)')

-- GPU/CPU
cmd:option('-gpu',0,'1=use gpu; 0=use cpu;')
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

params.vocab={} -- word frequency map
params.index2word={}
params.word2index={}

-- build the vocabulary
utils.buildVocab(params)

-- load data to main memory
utils.loadDataTensors(params) 

-- define the model
params.model=nn.Sequential()
if params.model_type=='rnn' then
	params.rnn=nn.Recurrent(params.wdim,nn.LookupTable(#params.index2word,params.wdim),nn.Linear(params.wdim,params.wdim),nn.Sigmoid())
	params.model:add(nn.Sequencer(params.rnn))
elseif params.model_type=='lstm' then
	params.rnn=nn.FastLSTM(params.wdim,params.wdim)
	params.model:add(nn.Sequencer(nn.LookupTable(#params.index2word,params.wdim)))
	params.model:add(nn.Sequencer(nn.FastLSTM(params.wdim,params.wdim)))
end
params.model:add(nn.Sequencer(nn.Linear(params.wdim,2)))
params.model:add(nn.Sequencer(nn.SoftMax()))
if params.dropout==1 then
	params.model:add(nn.Sequencer(nn.Dropout(params.dropout_prob)))
end
params.criterion=nn.SequencerCriterion(nn.CrossEntropyCriterion())

-- train the model
local idx=torch.randperm(#params.train_input_tensors)
print('Training ...')
local start=sys.clock()
for epoch=1,params.max_epochs do
	print('Epoch '..epoch..' ...')
	local epoch_start=sys.clock()
	local epoch_loss=0
	local iteration=0
	for i=1,#params.train_input_tensors,params.batch_size do
		xlua.progress(i,#params.train_input_tensors)
		local bsize=math.min(#params.train_input_tensors,i+params.batch_size-1)-i+1
		for j=1,bsize do
			local id=idx[i+j-1]
			local input_tensor={params.train_input_tensors[id]}
			local target_tensor={params.train_target_tensors[id]}
			local output=params.model:forward(input_tensor)
			local err=params.criterion:forward(output,target_tensor)
			local gradOutput=params.criterion:backward(output,target_tensor)
			params.model:backward(input_tensor,gradOutput)
			params.model:updateParameters(params.learning_rate)
			params.model:zeroGradParameters()
			iteration=iteration+1
			epoch_loss=epoch_loss+err
		end		
	end
	xlua.progress(#params.train_input_tensors,#params.train_input_tensors)
	print(string.format("Epoch %d done in %.2f minutes. loss=%f\n\n",epoch,((sys.clock()-epoch_start)/60),(epoch_loss/iteration)))
end
print(string.format("Done in %.2f seconds.",sys.clock()-start))