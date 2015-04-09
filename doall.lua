require 'cutorch'
require 'torch'
require 'nn'
require 'optim'

ffi = require('ffi')

torch.setdefaulttensortype('torch.FloatTensor')
dofile 'data.lua'
dofile 'train.lua'
dofile 'main.lua'

main()