function train_model(model, criterion, data, labels, test_data, test_labels, opt)
    parameters, grad_parameters = model:getParameters()
    model:training()
    model = model:cuda()
    for epoch =1, opt.nEpochs do
        local order = torch.randperm(opt.nBatches)
        for batch =1, opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            local minibatch = data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, data:size(2)):clone():cuda()
            local minibatch_labels = labels:sub(opt.idx, opt.idx + opt.minibatchSize):clone()
            local minibatch_loss = criterion:forward(model:forward(minibatch), minibatch_labels)
            model:zeroGradParameters()
            model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
            clr = opt.learningRate
            parameters:add(-clr, grad_parameters)
            print("epoch: ", epoch, " batch: ", batch)
            collectgarbage()
        end
        local accuracy = test_model(model, test_data, test_labels, opt)
        print("epoch ", epoch, " error: ", accuracy)
    end
end


function test_model(model, data, labels, opt)
    
    model:evaluate()
    local err = 0
    for t =1, data:size()[1], opt.minibatchSize do
        local pred = model:forward(data[{{t, math.min(t+opt.minibatchSize, data:size()[1])},{},{},{}}]:cuda())
        local _, argmax = pred:max(2)
        err = err + torch.ne(argmax:double(), labels[{{t, math.min(t+opt.minibatchSize, data:size()[1])}}]:double()):sum() 
    end
    err = err / labels:size(1)

    return err
end