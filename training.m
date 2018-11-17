clear all;
close all;
clc;


%% constructtion du dataset:
dataset = loadDataSet();


%% le nombre de neurones par couche:
nbrCarac = size(dataset.train_obj,1);
nbrOutputs = size(dataset.train_lab,1);
%LS = [nbrCarac; 40; 40; 30; nbrOutputs]; 
LS = [nbrCarac; 10; nbrOutputs];


%% le nombre des couches:
L = size(LS,1);


%% initialiser W et B à par des valeurs entre -1 et 1
W = cell(L,1);
for i=2:L
    W{i} = 2*rand(LS(i),LS(i-1))-1;
end

B = cell(L,1);
for i=2:L
    B{i} = 2*rand(LS(i),1)-1;
end

%% le nombre des epochs
epochs = 80;

 
 %% le pas H
 H = 0.01;

 %% miniBatch
 miniBatch = 4;
 
%% training network
for j=1:epochs
    param = upDateEpoch(dataset.train_obj,W,B,LS,dataset.train_lab,H,miniBatch);
    W = param.W;
    B = param.B; 
end       






