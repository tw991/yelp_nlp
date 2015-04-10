function train_model(model, criterion, data, labels, test_data, test_labels, opt)
    parameters, grad_parameters = model:getParameters()
    if opt.cuda == 'True' then
        model = model:cuda()
        criterion = criterion:cuda()
    end
    -- optimization functional to train the model with torch's optim library
    local function feval(x) 
        local minibatch = data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, data:size(2)):clone()
        local minibatch_labels = labels:sub(opt.idx, opt.idx + opt.minibatchSize):clone()
        model:training()
        if opt.cuda == 'True' then
            minibatch_loss = criterion:forward(model:forward(minibatch:cuda()):cuda(), minibatch_labels:cuda())
        else
            minibatch_loss = criterion:forward(model:forward(minibatch), minibatch_labels)
        end
        model:zeroGradParameters()
        if opt.cuda == 'True' then
            model:backward(minibatch:cuda(), criterion:backward(model.output, minibatch_labels:cuda()))
        else
            model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
        end
        return minibatch_loss, grad_parameters
    end
    
    for epoch=1,opt.nEpochs do
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
            print("epoch: ", epoch, " batch: ", batch)
        end
        collectgarbage()
        local accuracy = test_model(model, test_data, test_labels, opt)
        print("epoch ", epoch, " error: ", accuracy)

    end
end

function test_model(model, data, labels, opt)

    model:evaluate()
    local err = 0
    for t =1, data:size()[1], opt.minibatchSize do
        local input = data[{{t, math.min(t+opt.minibatchSize, data:size()[1])},{},{},{}}]
        if opt.cuda == 'True' then
            input = input:cuda()
        end
        local pred = model:forward(input)
        local _, argmax = pred:max(2)
        err = err + torch.ne(argmax:double(), labels[{{t, math.min(t+opt.minibatchSize, data:size()[1])}}]:double()):sum() 
    end
    err = err / labels:size(1)

    return err
end