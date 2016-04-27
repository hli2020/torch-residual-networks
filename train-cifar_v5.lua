require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'nngraph'

require 'residual-layers'
require 'train-helpers'
require 'data.cifar-dataset'

local nninit = require 'nninit'
-- nngraph.setDebug(true)
function stop() os.exit() end

-- hyli: new here, will be moved into modules in future update
function escapeCSV (s)
  if string.find(s, '[,"]') then
    s = '"' .. string.gsub(s, '"', '""') .. '"'
  end
  return s
end

function toCSV (tt)
  local s = ""
-- ChM 23.02.2014: changed pairs to ipairs 
-- assumption is that fromCSV and toCSV maintain data as ordered array
  for _,p in ipairs(tt) do  
    s = s .. "," .. escapeCSV(p)
  end
  return string.sub(s, 2)      -- remove first comma
end

function saveToCSV(log, name)
  -- log is a table
  -- assert(log ~= nil)
  -- print('passed!')
  fid = io.open(name..'.csv', "w")
  for _, row in ipairs(log) do
    fid:write(toCSV(row))
  end
  fid:close()
  os.execute("mv *.csv snapshots/" .. opt.expRootName .."/".. opt.note.."/"..timestamp)
end
----------

--[[
  v4 (exclusively for kevin, for ICSPCC and for PR course project)
  update: 
    1. change from Amazon S3 to local savings. lossLog, errorLog still unchanged :(
        -- save the log in a table (.t7 file) 
    2. change saving to best model scheme

  usage:
    type in the terminal:
            choice a (not recommended): GLOG_logtostderr=1 th train-cifar_v4.lua 2>&1 | tee shit.log 
            choide b: th train-cifar_v4.lua
]]

-- when debug mode is set, some intermediate
-- results (loss, shape, etc) will appear in the terminal.
local DEBUG = true
-- using AWS is good, but make sure the machine is connected to a STABLE connection
local AWS = false

opt = {
  plain_net         = true,       
  batchSize         = 128,
  iterSize          = 1,
  Nsize             = 18, --3, --18
  -- dataRoot          = "/home/zhizhen/cifar10torchsmall/cifar-10-batches-t7",
  dataRoot          = "/home/hongyang/dataset/cifar-10-batches-t7",
  -- dataRoot	    = "/media/DATADISK/hyli/dataset/cifar-10-batches-t7",
  loadFrom          = "",
  expRootName       = "cifar_ablation",
  expSuffix         = "server",
  -- indicate which machine it's deployed, 'ls139', 'fuck', whatever you like        
  gpuId             = 1   -- start from 1
  -- localSaveInterval = 2   -- in unit of epoch, DEPRECATED from v4
}

-- sdg init
sgdState = {
   learningRate   = "will be set later",    -- REMEMBER to check the lr_policy below
   weightDecay    = 0.0001,
   momentum       = 0.9,
   dampening      = 0,
   nesterov       = true,
   maxEpoch       = 200,
}

function get_lr(epoch)
  if epoch < 80 then
      sgdState.learningRate = opt.Nsize == 33 and 0.5 or 0.1
  elseif epoch < 120 then
      sgdState.learningRate = opt.Nsize == 33 and 0.005 or 0.01
  else
      sgdState.learningRate = opt.Nsize == 33 and 0.0005 or 0.001
  end
end

lossLog_local = {}
errorLog_local = {}
opt.beginToSave = 5
bestTop1 = 0
firstSave = true -- trivial variable

opt.note = string.format("depth_%d_bs%d_is%d", (6*opt.Nsize+2), 
    opt.batchSize, opt.iterSize)
if expSuffix ~= "" then
  opt.note = opt.note .. "_" .. opt.expSuffix
end

if not hasWorkbook then
  timestamp = os.date("%Y_%m_%d_%H:%M:%S")
else
  timestamp = workbook.tag
end

------- Feel free to comment these out -------
----------------------------------------------
hasWorkbook = false
if AWS then
  hasWorkbook, labWorkbook = pcall(require, 'lab-workbook')  
  if hasWorkbook then  
    workbook = labWorkbook:newExperiment{}
    lossLog = workbook:newTimeSeriesLog("Training loss", {"nImages", "loss", "lr"}, 100)
    errorLog = workbook:newTimeSeriesLog("Testing Error", {"nImages", "error", "loss_val"})
    workbook:saveGitStatus()
    workbook:saveJSON("opt", opt)
  else
    print "WARNING: No workbook support. No results will be saved in S3."
  end
end
----------------------------------------------
----------------------------------------------
-- make folder to hold local model results
os.execute("mkdir -p snapshots/"..opt.expRootName.."/"
  ..opt.note.."/"..timestamp)

print("Training settings:")
print(opt)
opt.gpuId = opt.gpuId or 1;
print("Running on GPU #", opt.gpuId)
cutorch.setDevice(opt.gpuId)

-- create data loader
dataTrain = Dataset.CIFAR(opt.dataRoot, "train", opt.batchSize)
dataTest = Dataset.CIFAR(opt.dataRoot, "test", opt.batchSize)
local mean,std = dataTrain:preprocess()
dataTest:preprocess(mean,std)
print("Training dataset size: ", dataTrain:size())
print("Testing dataset size: ", dataTest:size())
print('')

-- Residual network. Define the net in 'model'
-- Input: 3x32x32
local N = opt.Nsize
if opt.loadFrom == "" then

    input = nn.Identity()()
    -- model = nn.Sequential()
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
    print("output of random init: ")
    print(temp[{ {1}, {} }])

    -- save the network in local
    -- TODO: save it to S3
    -- update by hyli: N=18 failed to generate the net. try smaller depth 
    print('network graph saved (as .svg)!')
    graph.dot(model.fg, 'Forward Graph', 'network_graph')
    local command = string.format("mv network_graph.* snapshots/%s/%s/%s", 
     opt.expRootName, opt.note, timestamp)
    os.execute(command)
else
    print("Loading model from "..opt.loadFrom)
    model = torch.load(opt.loadFrom)
    print "Done"
end

-- stop()

loss = nn.ClassNLLCriterion()
loss:cuda()

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
    get_lr(sgdState.epochCounter)

    local lossTrain = 0
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
        lossTrain = lossTrain + loss:forward(y, labels)
        --print('first label is '..labels[1]..'.')
        --print(y[{{1}, {}}])     
        local df_dw = loss:backward(y, labels)
        model:backward(inputs, df_dw)

    end
    lossTrain = lossTrain / N
    gradients:mul( 1.0 / N )
    
    if DEBUG and (sgdState.nSampledImages%(10*opt.batchSize) ==  0) then 
        print(string.format('loss is %.3f', lossTrain))
    end

    if hasWorkbook then
      lossLog{nImages = sgdState.nSampledImages, loss = lossTrain, lr = sgdState.learningRate}
    end
    table.insert(lossLog_local, {})
    lossLog_local[#lossLog_local][1] = sgdState.nSampledImages
    lossLog_local[#lossLog_local][2] = lossTrain
    lossLog_local[#lossLog_local][3] = sgdState.learningRate
    lossLog_local[#lossLog_local][4] = "\n"

    -- the last argument is batchProcessed (aka, nSampledImages in sgd)
    return lossTrain, gradients, inputs:size(1) * N
end

function evalModel()

    local results = evaluateModel(model, dataTest, opt.batchSize)
    print(' * Current test accuracy top1:', results.correct1)
    print(string.format(' * Current test loss: %.3f', results.loss_val))

    local iter = sgdState.epochCounter
    if hasWorkbook then
      errorLog{ nImages = sgdState.nSampledImages or 0, 
                error = 1.0 - results.correct1, loss_val = results.loss_val }
      -- from v3, we dont save them to s3     
      -- if (iter or -1) % 100 == 0 then
      --   workbook:saveTorch("model", model)
      --   workbook:saveTorch("sgdState", sgdState)
      -- end
    end
    table.insert(errorLog_local, {})
    errorLog_local[#errorLog_local][1] = sgdState.nSampledImages
    errorLog_local[#errorLog_local][2] = 1.0 - results.correct1
    errorLog_local[#errorLog_local][3] = results.loss_val
    errorLog_local[#errorLog_local][4] = "\n"

    -- save the best model to local
    if ( iter >= opt.beginToSave) then

      if (results.correct1 > bestTop1) then

        bestTop1 = results.correct1
        bestEpoch = iter

        -- first delete previous best models
        if firstSave then
          firstSave = false
        else
          os.execute("rm snapshots/" .. opt.expRootName .."/".. opt.note.."/"
            ..timestamp.."/best_*")
        end

        torch.save(string.format("best_model_epoch_%d.t7", iter), model)
        torch.save(string.format("best_sgdState_epoch_%d.t7", iter), sgdState)
        os.execute("mv *.t7 snapshots/" .. opt.expRootName .."/".. opt.note.."/"..timestamp)
        -- torch.save(string.format("log_train_test_epoch_%d.t7", iter), LOG)
        -- fid = torch.DiskFile(string.format("log_train_test_epoch_%d.t7", iter), 'w')
        -- fid:writeObject(LOG)
        -- fid:close()
        -- assert(lossLog_local ~= nil)
        saveToCSV(lossLog_local, 'train_loss')
        saveToCSV(errorLog_local, 'test_error')

        print(' * SAVING BEST MODEL (model, optState, log) to local mahine...')
      end
      print(' * Best top1: '..bestTop1..', at epoch: '..bestEpoch)
    end
    
    if (hasWorkbook and sgdState.epochCounter == opt.beginToSave) then
      -- only excute once
      workbook:saveJSON("draw", {})
    end
    if (sgdState.epochCounter or 0) >= sgdState.maxEpoch then
      if hasWorkbook then
        workbook:saveJSON("done", {})
      end

      saveToCSV(lossLog_local, 'train_loss')
      saveToCSV(errorLog_local, 'test_error')
      print("Training done! Check the results!")
      os.exit()
    end
end

--[[
require 'graph'
graph.dot(model.fg, 'MLP', '/tmp/MLP')
os.execute('convert /tmp/MLP.svg /tmp/MLP.png')
display.image(image.load('/tmp/MLP.png'), {title="Network Structure", win=23})
--]]

-- dataTrain:size() is the epochSize
--------- Actual Training ----------
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
