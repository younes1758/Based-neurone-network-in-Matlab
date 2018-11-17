
YChap = testNetwork(dataset.test_obj(:,:),W,B,LS);

%mettre à 1 les elements ayants YLab(i) maximum
readDigits = size(YChap,2);
YClass = zeros(size(YChap,1),size(YChap,2));
for i=1:readDigits
    tmp = max(YChap(:,i));
    for j=1:LS(L)
        if(YChap(j,i) == tmp) 
            YClass(j,i) = 1;
        end
    end
end


curacy = 0;
for i=1:readDigits
    if(YClass(:,i) == dataset.test_lab(:,i))
        curacy = curacy +  1;
    end
end

curacy = curacy /readDigits






