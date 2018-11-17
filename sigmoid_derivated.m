function ret = sigmoid_derivated(vec)
% ret = sigmoid_derivated(vec) return the 
% segmoid derivated values of the vector vec

    ret = zeros(size(vec,1), size(vec,2));    
    ret = sigmf(vec,[1 0]).*(ones(size(vec,1), size(vec,2))-sigmf(vec,[1 0]));
   
end