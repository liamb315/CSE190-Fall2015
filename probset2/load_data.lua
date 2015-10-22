local mnist = {}

function mnist.download()
	tar = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'

	if not paths.dirp('mnist.t7') then
	   os.execute('wget ' .. tar)
	   os.execute('tar xvf ' .. paths.basename(tar))
	end	
end

function mnist.load_train()
	train_file = 'mnist.t7/train_32x32.t7'
	train_data = torch.load(train_file,'ascii')
	return train_data
end	


function mnist.load_test()
	test_file = 'mnist.t7/test_32x32.t7'
	test_data = torch.load(test_file,'ascii')	
	return test_data
end

return mnist


