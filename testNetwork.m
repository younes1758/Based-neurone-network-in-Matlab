function Y = testNetwork(X,W,B,LS)

    %X is a matrix of test dataset
    %Y is a matrix of labels calculated of the test dataset
    %W and B are lists of weigths and bias of the network
    %LS (Layauts Sizes) is a vector of layauts sizes.
    
    
    % L : nbr of layers
    L = size(LS,1);
        
    %sizeDataset : nbr of testing objects 
    sizeDataset = size(X,2);
    
    % 1 )initialise the list of matrice of weigth    

    A = cell(L,1);
    for i=1:L
        A{i} = zeros(LS(i),1);
    end

% 2 )initialise the list of the vectors of the linear transformation : Z
    Z = cell(L,1);
    for i=1:L
        Z{i} = zeros(LS(i),1);
    end

% 3 ) initialise the matrix of labels Y
    Y = zeros(LS(L),sizeDataset);
    
% 4 ) calculate the labels Y
     for j=1:sizeDataset
         
        %input activation
        A{1} = X(:,j);

        %feedForword
        for i=2:L
            Z{i} = mTimes(W{i},A{i-1}) + B{i};
            A{i} = sigmf(Z{i}, [1 0]);
        end
        
        Y(:,j) = A{L};
    end
end