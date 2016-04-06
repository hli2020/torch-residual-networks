require 'residual-layers'
require 'nn'
require 'data.cifar-dataset'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'train-helpers'
local nninit = require 'nninit'

--opt
opt = {
  batchSize         = 64,
  iterSize          = 2,
  Nsize             = 3,
  dataRoot          = "/home/hongyang/Desktop/cifar-10-batches-t7",
  loadFrom          = "",
  expRootName       = "cifar_ablation",
  expSuffix         = "",
  gpuId             = 1
}
opt.note = string.format("N_%d_size_%dx%d_%s", opt.Nsize, opt.batchSize, 
	opt.iterSize, opt.expSuffix)
print(opt.note)
print(opt.expSuffix==" ")

-- h1 = nn.Linear(20, 10)()
-- h3 = nn.Tanh()(h1)
-- h4 = nn.Linear(10, 10)(h3)
-- h5 = nn.Tanh()(h4)
-- h2 = nn.Linear(10, 1)(h5)
-- --h2 = nn.Linear(10, 1)(nn.Tanh()(nn.Linear(10, 10)(nn.Tanh()(h1))))
-- mlp = nn.gModule({h1}, {h2})

-- x = torch.rand(20)
-- dx = torch.rand(1)
-- mlp:updateOutput(x)
-- mlp:updateGradInput(x, dx)
-- mlp:accGradParameters(x, dx)

-- -- draw graph (the forward graph, '.fg')
-- graph.dot(mlp.fg, 'MLP')


-- h1 = nn.Linear(20, 20)()
-- h2 = nn.Linear(10, 10)()
-- hh1 = nn.Linear(20, 1)(nn.Tanh()(h1))
-- hh2 = nn.Linear(10, 1)(nn.Tanh()(h2))
-- madd = nn.CAddTable()({hh1, hh2})
-- oA = nn.Sigmoid()(madd)
-- oB = nn.Tanh()(madd)
-- gmod = nn.gModule({h1, h2}, {oA, oB})

-- x1 = torch.rand(20)
-- x2 = torch.rand(10)

-- gmod:updateOutput({x1, x2})
-- gmod:updateGradInput({x1, x2}, {torch.rand(1), torch.rand(1)})
-- graph.dot(gmod.fg, 'Big MLP')


-- input = nn.Identity()()
-- L1 = nn.Tanh()(nn.Linear(10, 20)(input))
-- L2 = nn.Tanh()(nn.Linear(30, 60)(nn.JoinTable(1)({input, L1})))
-- L3 = nn.Tanh()(nn.Linear(80, 160)(nn.JoinTable(1)({L1, L2})))

-- g = nn.gModule({input}, {L3})

-- indata = torch.rand(10)
-- gdata = torch.rand(160)
-- g:forward(indata)
-- g:backward(indata, gdata)

-- graph.dot(g.fg, 'Forward Graph')
-- graph.dot(g.bg, 'Backward Graph')

skip = nn.Padding(2, 16)
res = skip:forward(torch.rand(100, 16, 18, 18))

input = torch.rand(100, 32, 18, 18)
print(#input)
--assert(#input == #res)
print(#res)