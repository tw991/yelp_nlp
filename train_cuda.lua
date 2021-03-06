function train_model()
    err = 0
    model:training()
    epoch = epoch or 1
    parameters, grad_parameters = model:getParameters()
    -- optimization functional to train the model with torch's optim library
    order = torch.randperm(opt.nBatches)
    for batch =1, opt.nBatches do
        opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
        local minibatch = training_data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, training_data:size(2)):cuda()
        local minibatch_labels = training_labels:sub(opt.idx, opt.idx + opt.minibatchSize):cuda()
        grad_parameters:zero()
        local output = model:forward(minibatch)
        local _, argmax = output:max(2)
        err = err + torch.ne(argmax:double(), minibatch_labels:double()):sum() 
        local minibatch_loss = criterion:forward(output, minibatch_labels)
        model:backward(minibatch, criterion:backward(output, minibatch_labels))
        clr = opt.learningRate / (1+epoch * opt.learningRateDecay)
        parameters:add(-clr, grad_parameters)
        print("epoch: ", epoch, " batch: ", batch)
        collectgarbage()
    end
    local accuracy = test_model(training_data, training_labels)
    print("Training set: epoch ", epoch, " error: ", accuracy)
    local accuracy1 = test_model(test_data, test_labels)
    print("Testing set: epoch ", epoch, " error: ", accuracy1)
    epoch = epoch +1
end


function test_model(test_data, test_labels)

    model:evaluate()
    local err = 0
    for t =1, test_data:size()[1], opt.minibatchSize do
        local input = test_data[{{t, math.min(t+opt.minibatchSize, test_data:size()[1])},{},{},{}}]
        input = input:cuda()
        local pred = model:forward(input)
        local _, argmax = pred:max(2)
        err = err + torch.ne(argmax:double(), test_labels[{{t, math.min(t+opt.minibatchSize, test_data:size()[1])}}]:double()):sum() 
    end
    err = err / test_labels:size(1)

    return err
end