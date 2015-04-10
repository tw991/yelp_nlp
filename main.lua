function main()

    -- Configuration parameters
    opt = {}
    -- change these to the appropriate data locations
    opt.glovePath = "../glove_50.txt" -- path to raw glove data .txt file
    opt.dataPath = "../train.t7b"
    -- word vector dimensionality
    opt.inputDim = 50 
    -- nTrainDocs is the number of documents per class used in the training set, i.e.
    -- here we take the first nTrainDocs documents from each class as training samples
    -- and use the rest as a validation set.
    opt.nTrainDocs = 10000
    opt.nTestDocs = 0
    opt.nClasses = 5
    -- SGD parameters - play around with these
    opt.nEpochs = 100
    opt.minibatchSize = 128
    opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1
    opt.len = 100
    opt.cuda = 'True'

    print("Loading word vectors...")
    glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Loading raw data...")
    raw_data = torch.load(opt.dataPath)
 
    print("Computing document input representations...")
    processed_data, labels = preprocess_data(raw_data, glove_table, opt)
    
    -- split data into makeshift training and validation sets
    training_data = processed_data:sub(1, opt.nClasses*opt.nTrainDocs, 1, processed_data:size(2)):clone():reshape(opt.nClasses*opt.nTrainDocs, 1, 50, opt.len)
    training_labels = labels:sub(1, opt.nClasses*opt.nTrainDocs):clone()
    
    -- make your own choices - here I have not created a separate test set
    test_data = training_data:clone():float() 
    test_labels = training_labels:clone():float()

    raw_data =nil
    glove_table = nil
    -- construct model:
    model = nn.Sequential()
    -- if you decide to just adapt the baseline code for part 2, you'll probably want to make this linear and remove pooling
    model:add(nn.SpatialConvolution(1, 256, 3, 50, 1,1))
    model:add(nn.SpatialMaxPooling(2, 1, 2, 1))
    model:add(nn.SpatialConvolution(256,256, 4, 1, 1, 1))
    model:add(nn.SpatialMaxPooling(2,1,2,1))
    model:add(nn.SpatialConvolution(256,256, 2, 1, 1, 1))
    model:add(nn.SpatialMaxPooling(2,1,2,1))
    model:add(nn.Reshape(256*11, true))

    model:add(nn.Linear(256*11, 256))
    model:add(nn.ReLU())
    model:add(nn.Linear(256, 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
   
    epoch = 1   
    for i =1, opt.nEpochs do
        train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
    end
    local results = test_model(model, test_data, test_labels)
    print(results)
end
