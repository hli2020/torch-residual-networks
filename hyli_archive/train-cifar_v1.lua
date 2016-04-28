--[[
Copyright (c) 2016 Michael Wilber
--]]

require 'residual-layers'
require 'nn'
require 'data.cifar-dataset'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'train-helpers'
local nninit = require 'nninit'
nngraph.setDebug(true)

-- Feel free to comment these out.
hasWorkbook, labWorkbook = pcall(require, 'lab-workbook')
if hasWorkbook then
  workbook = labWorkbook:newExperiment{}
  lossLog = workbook:newTimeSeriesLog("Training loss",
                                      {"nImages", "loss"},
                                      100)
  errorLog = workbook:newTimeSeriesLog("Testing Error",
                                       {"nImages", "error"})
else
  print "WARNING: No workbook support. No results will be saved."
end

opt = lapp[[
      --batchSize       (default 64)       Sub-batch size
      --iterSize        (default 2)         How many sub-batches in each batch
      --Nsize           (default 18)         Model has 6*n+2 layers.
      --dataRoot        (default "/home/hongyang/Desktop/cifar-10-batches-t7") Data root folder
      --loadFrom        (default "")        Model to load
      --experimentName  (default "snapshots/cifar-residual-experiment1")      
      --gpuId           (default 1)
      --note            (default "N_18_iter_size_2")
]]
print("Training settings:")
print(opt)
cutorch.setDevice(opt.gpuId)

-- create data loader
dataTrain = Dataset.CIFAR(opt.dataRoot, "train", opt.batchSize)
dataTest = Dataset.CIFAR(opt.dataRoot, "test", opt.batchSize)

-- -- fucking debug
-- dataRoot = "/home/hongyang/Desktop/cifar-10-batches-t7"
-- batchSize = 128
-- dataTrain = Dataset.CIFAR(dataRoot, "train", batchSize)
-- collectgarbage();
-- collectgarbage();
-- require 'image'
-- image.display(dataTrain:sample(32).inputs)

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

    -- got crazy nan outputs
    local aa = model:forward(torch.randn(100, 3, 32,32):cuda())
    print(aa[{ {1}, {} }])
    graph.dot(model.fg, 'Forward Graph', opt.note)

else
    print("Loading model from "..opt.loadFrom)
    model = torch.load(opt.loadFrom)
    print "Done"
end

loss = nn.ClassNLLCriterion()
loss:cuda()

sgdState = {
   -- My semi-working settings
   learningRate   = "will be set later",
   weightDecay    = 0.0001,
   -- Settings from their paper
   --learningRate = 0.1,
   --weightDecay    = 1e-4,

   momentum     = 0.9,
   dampening    = 0,
   nesterov     = true,
}

if opt.loadFrom ~= "" then
    print("Trying to load sgdState from "..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    sgdState = torch.load(""..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    print("Got", sgdState.nSampledImages,"images")
end

-- Begin saving the experiment to our workbook
if hasWorkbook then
  workbook:saveGitStatus()
  workbook:saveJSON("opt", opt)
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
    for i=1,N do
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
    
    -- if sgdState.nSampledImages%(100*opt.batchSize) ==  0 then 
    --     print(string.format('loss is %.3f', loss_val))
    -- end

    gradients:mul( 1.0 / N )

    if hasWorkbook then
      lossLog{nImages = sgdState.nSampledImages,
              loss = loss_val}
    end

    -- the last argument is batchProcessed (aka, nSampledImages in sgd)
    return loss_val, gradients, inputs:size(1) * N
end

function evalModel()
    -- input is dataTest to model
    local results = evaluateModel(model, dataTest, opt.batchSize)
    --print(string.format(' * Test accuracy top1: %.3f  top5: %.3f', results.correct1, results.correct5))
    print(' * Test accuracy top1:', results.correct1)

    if hasWorkbook then
      errorLog{nImages = sgdState.nSampledImages or 0,
               error = 1.0 - results.correct1}
      if (sgdState.epochCounter or -1) % 10 == 0 then
        workbook:saveTorch("model", model)
        workbook:saveTorch("sgdState", sgdState)
      end
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

--[[
require 'ncdu-model-explore'
local y = model:forward(torch.randn(opt.batchSize, 3, 32,32):cuda())
local df_dw = loss:backward(y, torch.zeros(opt.batchSize):cuda())
model:backward(torch.randn(opt.batchSize,3,32,32):cuda(), df_dw)
exploreNcdu(model)
--]]

-- dataTrain:size() is the epochSize
-- Actual Training! -----------------------------
weights, gradients = model:getParameters()

TrainingHelpers.trainForever(
forwardBackwardBatch,
weights,
sgdState,
dataTrain:size(),
evalModel
)