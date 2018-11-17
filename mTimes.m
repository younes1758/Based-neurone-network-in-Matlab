function ret = mTimes(m1, m2)
    n = size(m1,1);
    m = size(m2,2);
    l = size(m1,2);
    ret = zeros(n,m);
    for i=1:n
        for j=1:m
            for k=1:l
                ret(i,j) = ret(i,j)+m1(i,k)*m2(k,j);
            end
        end
    end
end