
function stop() return os.exit() end

-- -- creates an array which contains twice the same tensor
-- array = {}
-- x = torch.Tensor(1)
-- table.insert(array, x)
-- table.insert(array, x)

-- -- array[1] and array[2] refer to the same address
-- -- x[1] == array[1][1] == array[2][1] == 3.14
-- array[1][1] = 3.14

-- -- write the array on disk
-- file = torch.DiskFile('foo.asc', 'w')
-- file:writeObject(array)
-- file:close() -- make sure the data is written

-- stop()

-- reload the array
-- file = torch.DiskFile('foo.asc', 'r')
-- arrayNew = file:readObject()

-- print(arrayNew)
-- -- arrayNew[1] and arrayNew[2] refer to the same address!
-- -- arrayNew[1][1] == arrayNew[2][1] == 3.14
-- -- so if we do now:
-- arrayNew[1][1] = 2.72
-- -- arrayNew[1][1] == arrayNew[2][1] == 2.72 !
-- print(arrayNew[2][1])


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

-- DEBUG
-- write data

fid = io.open('training_loss.csv', "w")

i = 1
log = {}
log2 = {}
while i < 10 do
   table.insert(log, {})
   table.insert(log2, {})
   log[#log][1] = 2000
   log[#log][2] = 2.7
   log[#log][3] = 0.001
   log[#log][4] = "\n"
   currRow = toCSV(log[#log])
   fid:write(currRow)

   log2[#log2][1] = 2000
   log2[#log2][2] = 0.22
   log2[#log2][3] = 3.4       --test loss

   i = i + 1
end
-- fid = torch.DiskFile('loss.t7', 'w')
-- fid:writeObject({log, log2})
-- fid:close()

fid:close()


-- failed
-- csv = Csv('haha', "w")
-- csv:write(log)
-- csv:close()

-- stop()

-- read data
-- fid = torch.DiskFile('loss.t7', 'r')
-- loss = fid:readObject()
-- print(#loss)
-- print(loss)
-- for _, row in ipairs(loss) do
--    print("row: "..row[1].." "..row[2].." "..row[3])
-- end
