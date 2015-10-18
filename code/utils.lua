--[[

Utility function used by lua classes.

--]]

local utils={}

-- Function to check if the input is a valid number
function utils.isNumber(a)
	if tonumber(a) ~= nil then
		return true
	end
	return false
end

-- Function to trim the string
function utils.trim(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

-- Function to split a string by given char.
function utils.splitByChar(str,inSplitPattern)
	outResults={}
	local theStart = 1
	local theSplitStart,theSplitEnd=string.find(str,inSplitPattern,theStart)
	while theSplitStart do
		table.insert(outResults,string.sub(str,theStart,theSplitStart-1))
		theStart=theSplitEnd+1
		theSplitStart,theSplitEnd=string.find(str,inSplitPattern,theStart)
	end
	table.insert(outResults,string.sub(str,theStart))
	return outResults
end

-- Function to pad tokens.
function utils.padTokens(tokens,pad)
	local res={}

	-- Append begin tokens
	for i=1,pad do
		table.insert(res,'<bpad-'..i..'>')
	end

	for _,word in ipairs(tokens) do
		word=utils.splitByChar(word,'%$%$%$')[1]
		table.insert(res,word)
	end

	-- Append end tokens
	for i=1,pad do
		table.insert(res,'<epad-'..i..'>')
	end

	return res
end

-- Function to get all ngrams
function utils.getNgrams(doc,n,pad)
	local res={}
	local tokens=utils.padTokens(utils.splitByChar(doc,'\t'),pad)
	for i=1,(#tokens-n+1) do
		local word=''
		for j=i,(i+(n-1)) do
			word=word..tokens[j]..' '
		end
		word=utils.trim(word)
		table.insert(res,word)
	end
	return res
end

-- Function to process a sentence to build vocab
function utils.processSentence(config,sentence)
	local pad=1
	for _,word in ipairs(utils.getNgrams(sentence,1,pad)) do
		config.total_count=config.total_count+1

		word=utils.splitByChar(word,'%$%$%$')[1]

		if config.to_lower==1 then
			word=word:lower()
		end

		-- Fill word vocab.
		if config.vocab[word]==nil then
			config.vocab[word]=1
		else
			config.vocab[word]=config.vocab[word]+1
		end
	end
	config.corpus_size=config.corpus_size+1
end

-- Function to build vocabulary from the corpus
function utils.buildVocab(config)
	print('Building vocabulary...')
	local start=sys.clock()
	local fptr=io.open(config.train_file,'r')
	
	-- Fill the vocabulary frequency map
	config.total_count=0
	config.corpus_size=0
	while true do
		local pid=fptr:read()
		if pid==nil then
			break
		end
		local title=fptr:read()
		utils.processSentence(config,title)
		local abstract=fptr:read()
		utils.processSentence(config,abstract)
	end
	fptr.close()

	-- Discard the words that doesn't meet minimum frequency and create indices.
	for word,count in pairs(config.vocab) do
		if count<config.min_freq then
			config.vocab[word]=nil
		else
			config.index2word[#config.index2word+1]=word
			config.word2index[word]=#config.index2word
		end
	end

	-- Add unknown word
	config.vocab['<UK>']=1
	config.index2word[#config.index2word+1]='<UK>'
	config.word2index['<UK>']=#config.index2word
	config.vocab_size= #config.index2word

	print(string.format("%d words, %d documents processed in %.2f seconds.",config.total_count,config.corpus_size,sys.clock()-start))
	print(string.format("Vocab size after eliminating words occuring less than %d times: %d",config.min_freq,config.vocab_size))
end

-- Function to extract input and target tensors
function utils.extractInputTargetTensor(config,data) 
	local content=utils.splitByChar(data,'\t')
	local input_tensor=torch.Tensor(#content)
	local target_tensor=torch.Tensor(#content)
	for i,entity in ipairs(content) do
		local word=utils.splitByChar(entity,'%$%$%$')[1]
		if config.word2index[word]~=nil then
			input_tensor[i]=config.word2index[word]
		else
			input_tensor[i]=config.word2index['<UK>']
		end
		local target=utils.splitByChar(entity,'%$%$%$')[4]
		if target=='O' then
			target_tensor[i]=1
		else
			target_tensor[i]=2
		end
	end
	return input_tensor,target_tensor
end

-- Function to extract input tensors
function utils.extractInputTensor(config,data) 
	local content=utils.splitByChar(data,'\t')
	local input_tensor=torch.Tensor(#content)
	for i,entity in ipairs(content) do
		local word=utils.splitByChar(entity,'%$%$%$')[1]
		if config.word2index[word]~=nil then
			input_tensor[i]=config.word2index[word]
		else
			input_tensor[i]=config.word2index['<UK>']
		end
	end
	return input_tensor
end

-- Function to load input and target tensors.
function utils.loadDataTensors(config) 
	-- load train set tensors
	config.train_input_tensors={}
	config.train_target_tensors={}
	local fptr=io.open(config.train_file,'r')
	while true do
		local pid=fptr:read()
		if pid==nil then
			break
		end
		local title=fptr:read()
		local input_tensor,target_tensor=utils.extractInputTargetTensor(config,title) 
		table.insert(config.train_input_tensors,input_tensor)
		table.insert(config.train_target_tensors,target_tensor)
		local abstract=fptr:read()
		local input_tensor,target_tensor=utils.extractInputTargetTensor(config,abstract) 
		table.insert(config.train_input_tensors,input_tensor)
		table.insert(config.train_target_tensors,target_tensor)
	end
	fptr.close()
	-- load dev set tensors
	config.dev_input_tensors={}
	config.dev_target_tensors={}
	local fptr=io.open(config.dev_file,'r')
	while true do
		local pid=fptr:read()
		if pid==nil then
			break
		end
		local title=fptr:read()
		local input_tensor,target_tensor=utils.extractInputTargetTensor(config,title) 
		table.insert(config.dev_input_tensors,input_tensor)
		table.insert(config.dev_target_tensors,target_tensor)
		local abstract=fptr:read()
		local input_tensor,target_tensor=utils.extractInputTargetTensor(config,abstract) 
		table.insert(config.dev_input_tensors,input_tensor)
		table.insert(config.dev_target_tensors,target_tensor)
	end
	fptr.close()
	-- load test set tensors
	config.test_input_tensors={}
	local fptr=io.open(config.test_file,'r')
	while true do
		local pid=fptr:read()
		if pid==nil then
			break
		end
		local title=fptr:read()
		local input_tensor=utils.extractInputTensor(config,title) 
		table.insert(config.test_input_tensors,input_tensor)
		local abstract=fptr:read()
		local input_tensor=utils.extractInputTensor(config,abstract) 
		table.insert(config.test_input_tensors,input_tensor)
	end
	fptr.close()
end

return utils