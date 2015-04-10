function train_model(model, criterion, data, labels, test_data, test_labels, opt)
    parameters, grad_parameters = model:getParameters()
    model = model:cuda()
    criterion = criterion:cuda()
    -- optimization functional to train the model with torch's optim library


    for epoch =1, opt.nEpochs do
        order = torch.randperm(opt.nBatches)
        for batch =1, opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            minibatch = data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, data:size(2)):clone()
            minibatch_labels = labels:sub(opt.idx, opt.idx + opt.minibatchSize):clone()
            model:training()
            minibatch_loss = criterion:forward(model:forward(minibatch:cuda()):cuda(), minibatch_labels:cuda())
            model:zeroGradParameters()
            model:backward(minibatch:cuda(), criterion:backward(model.output, minibatch_labels:cuda()))
            clr = opt.learningRate / (1+epoch * opt.learningRateDecay)
            parameters:add(-clr, grad_Parameters)
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
        input = input:cuda()
        local pred = model:forward(input)
        local _, argmax = pred:max(2)
        err = err + torch.ne(argmax:double(), labels[{{t, math.min(t+opt.minibatchSize, data:size()[1])}}]:double()):sum() 
    end
    err = err / labels:size(1)

    return err
end