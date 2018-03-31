clc
clear all

load "dataset/raw_dataset_workspace"

data(1, :)

P = data(:, 1:34);
T = data(:, 35);

Pn = mapminmax(P, -1, 1);

save normalized

[Ts, ind_Ts] = sort(T);
Pns = zeros(size(Pn));

for i = 1:length(ind_Ts)
    Pns(i, :) = Pn(ind_Ts(i), :);
endfor

clear i
clear data
clear ind_Ts
