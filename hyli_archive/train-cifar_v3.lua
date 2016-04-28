--[[
Copyright (c) 2016 Hongyang
--]]
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'nngraph'

require 'residual-layers'
require 'train-helpers'
require 'data.cifar-dataset'

local nninit = require 'nninit'

--nngraph.setDebug(true)
function stop() os.exit() end

--[[
  v3
]]
-- when debug mode is set, some intermediate
-- results (loss, shape, etc) will appear in the terminal.
local DEBUG = true
local AWS = false

opt = {
  batchSize         = 2,
  iterSize          = 64,
  Nsize             = 200,
  --dataRoot          = "/home/hongyang/Desktop/cifar-10-batches-t7",
  dataRoot	    = "/media/DATADISK/hyli/dataset/cifar-10-batches-t7",
  loadFrom          = "",
  expRootName       = "cifar_ablation",
  expSuffix         = "149_no_relu_1202_layer",        -- indicate which machine it's deployed
  gpuId             = 3,
  localSaveInterval = 50
}

opt.note = string.format("N_%d_size_b%d_i%d", opt.Nsize, 
    opt.batchSize, opt.iterSize)
if expSuffix ~= "" then
  opt.note = opt.note .. "_" .. opt.expSuffix
end
-- make folder to hold local model results
os.execute("mkdir -p snapshots/"..opt.expRootName.."/"..opt.note)

print("Training settings:")
print(opt)
opt.gpuId = opt.gpuId or 1;
print("Running on GPU #", opt.gpuId)
cutorch.setDevice(opt.gpuId)

------- Feel free to comment these out -------
----------------------------------------------
hasWorkbook = false
if AWS then
  hasWorkbook, labWorkbook = pcall(require, 'lab-workbook')  
  if hasWorkbook then  
    workbook = labWorkbook:newExperiment{}
    lossLog = workbook:newTimeSeriesLog("Training loss", {"nImages", "loss", "lr"}, 200)
    errorLog = workbook:newTimeSeriesLog("Testing Error", {"nImages", "error"})
    workbook:saveGitStatus()
    workbook:saveJSON("opt", opt)
  else
    print "WARNING: No workbook support. No results will be saved."
  end
end
----------------------------------------------
----------------------------------------------
-- a = workbook.tag
-- print(a)
-- stop()

-- create data loader
dataTrain = Dataset.CIFAR(opt.dataRoot, "train", opt.batchSize)
dataTest = Dataset.CIFAR(opt.dataRoot, "test", opt.batchSize)
local mean,std = dataTrain:preprocess()
dataTest:preprocess(mean,std)
print("Training dataset size: ", dataTrain:size())

-- Residual network. Define the net in 'model'
-- Input: 3x32x32
local N = opt.Nsize
if opt.loadFrom == "" then

    input = nn.Identity()()
    --print(input)    -- nngraph.Node
    ------> 3, 32,32
    model = cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
                :init('weight', nninit.kaiming, {gain = 'relu'})
                :init('bias', nninit.constant, 0)(input)

    model = cudnn.SpatialBatchNormalization(16)
                                :init('weight', nninit.normal, 1, 0.002)
                                :init('bias', nninit.constant, 0)(model)
    model = cudnn.ReLU(true)(model)
    
    ------> 16, 32,32   First Group
    for i=1,N do   model = addResidualLayer2(model, 16)   end
    
    ------> 32, 16,16   Second Group
    model = addResidualLayer2(model, 16, 32, 2)
    for i=1,N-1 do   model = addResidualLayer2(model, 32)   end
    
    ------> 64, 8,8     Third Group
    model = addResidualLayer2(model, 32, 64, 2)
    for i=1,N-1 do   model = addResidualLayer2(model, 64)   end
    
    ------> 10, 8,8     Pooling, Linear, Softmax
    model = nn.SpatialAveragePooling(8,8)(model)
    model = nn.Reshape(64)(model)
    model = nn.Linear(64, 10)
                          :init('weight', nninit.normal, 0, 0.05)
                          :init('bias', nninit.constant, 0)(model)
    model = nn.LogSoftMax()(model)

    model = nn.gModule({input}, {model})
    model:cuda()

    -- got crazy nan outputs if initialized improperly
    local temp = model:forward(torch.randn(100, 3, 32,32):cuda())
    print("output of random init:")
    print(temp[{ {1}, {} }])

    -- save the network in local
    -- TODO: save it to S3
    graph.dot(model.fg, 'Forward Graph', 'net_architecture')
    local command = string.format("mv net_architecture.* snapshots/%s/%s", opt.expRootName, opt.note)
    os.execute(command)

else
    print("Loading model from "..opt.loadFrom)
    model = torch.load(opt.loadFrom)
    print "Done"
end

--stop()

loss = nn.ClassNLLCriterion()
loss:cuda()

-- init
sgdState = {
   -- My semi-working settings
   learningRate   = "will be set later",
   weightDecay    = 0.0001,
   momentum       = 0.9,
   dampening      = 0,
   nesterov       = true,
}

if opt.loadFrom ~= "" then
    print("Trying to load sgdState from "..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    sgdState = torch.load(""..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    print("Got", sgdState.nSampledImages,"images")
end

function forwardBackwardBatch(batch)
    -- After every batch, the different GPUs all have different gradients
    -- (because they saw different data), and only the first GPU's weights were
    -- actually updated.
    -- We have to do two changes:
    --   - Copy the new parameters from GPU #1 to the rest of them;
    --   - Zero the gradient parameters so we can accumulate them again.
    model:training()
    gradients:zero()

    --[[
    -- Reset BN momentum, nvidia-style
    model:apply(function(m)
        if torch.type(m):find('BatchNormalization') then
            m.momentum = 1.0  / ((m.count or 0) + 1)
            m.count = (m.count or 0) + 1
            print("--Resetting BN momentum to", m.momentum)
            print("-- Running mean is", m.running_mean:mean(), "+-", m.running_mean:std())
        end
    end)
    --]]

    -- From https://github.com/bgshih/cifar.torch/blob/master/train.lua#L119-L128
    if sgdState.epochCounter < 80 then
        sgdState.learningRate = 0.1
    elseif sgdState.epochCounter < 120 then
        sgdState.learningRate = 0.01
    else
        sgdState.learningRate = 0.001
    end

    local loss_val = 0
    local N = opt.iterSize
    local inputs, labels
    for i = 1, N do
        inputs, labels = dataTrain:getBatch()
        --inputs = torch.rand(128,3,32,32)
        --print(#inputs)
        inputs = inputs:cuda()
        labels = labels:cuda()
        collectgarbage(); collectgarbage();
        local y = model:forward(inputs)
        loss_val = loss_val + loss:forward(y, labels)

        --print('first label is '..labels[1]..'.')
        --print(y[{{1}, {}}])
        
        local df_dw = loss:backward(y, labels)
        model:backward(inputs, df_dw)
        -- The above call will accumulate all GPUs' parameters onto GPU #1
    end
    loss_val = loss_val / N
    gradients:mul( 1.0 / N )
    
    if DEBUG and (sgdState.nSampledImages%(10*opt.batchSize) ==  0) then 
        print(string.format('loss is %.3f', loss_val))
    end

    if hasWorkbook then
      lossLog{nImages = sgdState.nSampledImages, loss = loss_val, lr = sgdState.learningRate}
    end

    -- the last argument is batchProcessed (aka, nSampledImages in sgd)
    return loss_val, gradients, inputs:size(1) * N
end

function evalModel()

    local results = evaluateModel(model, dataTest, opt.batchSize)
    --print(string.format(' * Test accuracy top1: %.3f  top5: %.3f', results.correct1, results.correct5))
    print(' * Test accuracy top1:', results.correct1)

    local iter = sgdState.epochCounter
    if hasWorkbook then
      
      errorLog{ nImages = sgdState.nSampledImages or 0, 
                error = 1.0 - results.correct1 }

      -- from v3, we dont save them to s3     
      -- if (iter or -1) % 100 == 0 then
      --   workbook:saveTorch("model", model)
      --   workbook:saveTorch("sgdState", sgdState)
      -- end
    end

    -- save a copy to local
    if ( iter > 1) and (iter % opt.localSaveInterval == 0) then
      torch.save(string.format("model_epoch_%d.t7", iter), model)
      torch.save(string.format("sgdState_epoch_%d.t7", iter), sgdState)
      os.execute("mv *.t7 snapshots/" .. opt.expRootName .."/".. opt.note)
      print(' * saving model and optState to local mahine...')
    end

    if (sgdState.epochCounter or 0) > 200 then
        print("Training done! Check the results!")
        os.exit()
    end
end

--evalModel()

--[[
require 'graph'
graph.dot(model.fg, 'MLP', '/tmp/MLP')
os.execute('convert /tmp/MLP.svg /tmp/MLP.png')
display.image(image.load('/tmp/MLP.png'), {title="Network Structure", win=23})
--]]

-- dataTrain:size() is the epochSize
-- Actual Training! -----------------------------
weights, gradients = model:getParameters()

-- some interesting stuff here
-- results = TrainingHelpers.inspectModel(model)
-- TrainingHelpers.printInspection(results)

TrainingHelpers.trainForever(
  forwardBackwardBatch,
  weights,
  sgdState,
  dataTrain:size(),
  evalModel,
  hasWorkbook
)
