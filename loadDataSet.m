function dataset = loadDataSet()

% dataset is an strucrot that contains : 
%   train_obj : matrix of train'objects
%   train_lab : vector of train'labels
%   test_obj : matrix of test'objects
%   test_lab : vecotr of test'labels
%   matProjection : projection matrix
% after applying ACP.

    %% output layer
    nbrOutputs = 10;
    
    %% training dataset
    imgFile =  'dataset\train-images.idx3-ubyte';
    labelFile = 'dataset\train-labels.idx1-ubyte';
    readDigits_training = 100;
    offset = 0;
    [train_images, labels] = readMNIST(imgFile, labelFile, readDigits_training, offset);

    dataset.train_lab = zeros(nbrOutputs,readDigits_training);
    for i=1:readDigits_training
    dataset.train_lab(labels(i)+1,i) = 1;
    end
    
    %% test dataset
    imgFile = 'dataset\t10k-images.idx3-ubyte';
    labelFile = 'dataset\t10k-labels.idx1-ubyte';
    readDigits_test = 100;
    offset = 0;
    [test_images, test_labels] = readMNIST(imgFile, labelFile, readDigits_test, offset);
    
    dataset.test_lab = zeros(nbrOutputs,readDigits_test);
    for i=1:readDigits_test
    dataset.test_lab(test_labels(i)+1,i) = 1;
    end
    
    
    %% build the dataset for the PCA
    dataset_train = zeros(size(train_images,1)*size(train_images,2),size(train_images,3));
    dataset_test = zeros(size(test_images,1)*size(test_images,2),size(test_images,3));
    
    for i=1:size(train_images,3)
        dataset_train(:,i) = reshape(train_images(:,:,i)', [size(train_images,1)*size(train_images,2) 1]);
    end
    
    for i=1:size(test_images,3)
        dataset_test(:,i) = reshape(test_images(:,:,i)', [size(test_images,1)*size(test_images,2) 1]);
    end
    
    
    %% Apply ACP to train_images
    [coeff,score,latent] = pca(dataset_train','NumComponents',390);   
       
    
    %% projection of test_images in the new repere    
    Mean = mean(dataset_test,2);
    test_imagesCentred = dataset_test' - ones(size(dataset_test,2), 1) * Mean';
    
    %% build the output object    
    dataset.test_obj =  (test_imagesCentred * coeff)' ;
    dataset.train_obj = score';    
    dataset.matProjection = coeff;
    
end
