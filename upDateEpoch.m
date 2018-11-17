function param = upDateEpoch(X,W,B,LS,lab,H,miniBatch)

%X is an matrix pf training dataset

    % L : nbr of layers
    L = size(LS,1);   
    
    %sizeDataset : nbr of training objects 
    sizeDataset = size(X,2);
    
% 1 )initialise the list of activations    

    A = cell(L,1);
    for i=1:L
        A{i} = zeros(LS(i),1);
    end

% 2 )initialise the list of the vectors of the linear transformation : Z
    Z = cell(L,1);
    for i=1:L
        Z{i} = zeros(LS(i),1);
    end
    
% 3 ) initialise the list of the vectors of the errors of each layaut : 
    ER = cell(L,1);
    for i=1:L
        ER{i} = zeros(LS(i),1);
    end
    
    ER_times_A_Sum = cell(L,1);
    for i=1:L
        ER_times_A_Sum{i} = zeros(LS(i),1);
    end
    
    ER_sum{L} =  cell(L,1);
    for i=1:L
      ER_sum{i} = zeros(LS(i),1);
    end

    %counteur pour conter le nombre des images pour faire le mise à jour
    counter = 0;
    
% 4 ) compute the weigths and bias and errors of each neuron using all mini-batch tuples  
    for j=1:sizeDataset
        counter = counter + 1;
        %input activation
        A{1} = X(:,j);
        
        %output desired when A{1} is input
        Y = lab(:,j);
        
        %feedForword
        for i=2:L
            Z{i} = mTimes(W{i},A{i-1}) + B{i};
            A{i} = sigmf(Z{i}, [1 0]);
        end

        %backword 
        ER{L} = (A{L}-Y).*sigmoid_derivated(Z{L});
        ER_sum{L} = ER_sum{L} + ER{L};
        ER_times_A_Sum{L} = ER_times_A_Sum{L} + ER{L}*(A{L-1})';         

        for i=L-1:-1:2
            ER{i} = ((W{i+1})'*ER{i+1}).*sigmoid_derivated(Z{i});            
            ER_sum{i} = ER_sum{i} + ER{i};
            ER_times_A_Sum{i} = ER_times_A_Sum{i} + ER{i}*(A{i-1})';
        end
                 
        if(counter == miniBatch)
            
            counter = 0;
            
            % 5 ) update weigths and bias:
            for i=2:L
                W{i} = W{i} - (H/miniBatch) * ER_times_A_Sum{i};
                B{i} = B{i} - (H/miniBatch) * ER_sum{i};    
            end
            
            % 6 ) reinitialise the list of the vectors of the errors of each layaut : 
            for i=1:L
                ER{i} = zeros(LS(i),1);
                ER_times_A_Sum{i} = zeros(LS(i),1);
                ER_sum{i} = zeros(LS(i),1);
            end
            
        end
    end
    
    %return final weigths and bias
    for i=2:L
        param.W{i} = W{i} ;
        param.B{i} = B{i} ;   
    end
    
end