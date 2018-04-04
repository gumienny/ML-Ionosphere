clc
clear all
close all
format compact

load "dataset/raw_dataset_workspace"

T = data(:, 35);
P = mapminmax(data(:, 1:34), -1, 1); % P jest znormalizowane

% save normalized

% [Ts, ind_Ts] = sort(T);
% Pns = zeros(size(Pn));

% for i = 1:length(ind_Ts)
%     Pns(i, :) = Pn(ind_Ts(i), :);
% end

P = P';
T = T';

S1_vec     = [10:10:100];
S2_vec     = S1_vec;
lr_inc_vec = [1.01:0.02:1.09];
lr_dec_vec = [0.5:0.1:0.9];
er_vec     = [1.02:0.01:1.06];

liczba_petli = length(S1_vec) * length(S2_vec) * length(lr_inc_vec) * length(lr_dec_vec) * length(er_vec);
ind_petli    = 0;
disp_freq    = 100;
max_epoch    = 100;
err_goal     = 0.25;
lr           = 1e-5;
qmax         = 0;

for ind_S1 = 1:length(S1_vec),
    for ind_S2 = 1:ind_S1,
        [S3, Q] = size(T);

        %  INITFF
        %  [W1,B1,...] = INITFF(P,S1,'F1',...,Sn,'Fn')
        %    P  - Rx2 matrix of input vectors.
        %    Si - Size of ith layer.
        %    Fi - Transfer function of the ith layer (string).
        %  Returns weights and biases:
        %    Wi - Weight matrix of the ith layer.
        %    Bi - Bias (column) vector of the ith layer.

        [W1, B1, W2, B2, W3, B3] = initff(P, S1_vec(ind_S1), 'tansig', S2_vec(ind_S2), 'tansig', S3, 'purelin');

        for ind_lr_inc = 1:length(lr_inc_vec),
            for ind_lr_dec = 1:length(lr_dec_vec),
                for ind_er = 1:length(er_vec),
                    %  TRAINBPA Train feed-forward network with bp + adaptive learning.
                    %  Can be called with 1, 2, or 3 sets of weights
                    %  and biases to train up to 3 layer feed-forward networks.
                    %          
                    %  [W1,B1,W2,B2,...,TE,TR] = TRAINBPA(W1,B1,F1,W2,B2,F2,...,P,T,TP)
                    %    Wi - SixR weight matrix for the ith layer.
                    %    Bi - S1x1 bias vector for the ith layer.
                    %    Fi - Transfer function (string) for the ith layer.
                    %    P  - RxQ matrix of input vectors.
                    %    T  - SxQ matrix of target vectors.
                    %    TP - Training parameters (optional).
                    %
                    %  Returns new weights and biases and
                    %    Wi - new weights.
                    %    Bi - new biases.
                    %    TE - the actual number of epochs trained.
                    %    TR - training record: [row of errors]
                    %  
                    %  Training parameters are:
                    %    TP(1) - Ilość epok między wyświetlaniem stanu, default = 25.
                    %    TP(2) - Maksymalna liczba trenowanych epok,    default = 1000.
                    %    TP(3) - Dopuszczalny błąd SSE,                 default = 0.02.
                    %    TP(4) - Współczynnik uczenia,                  0.01.
                    %    TP(5) - Wzrost szubkości uczenia,              default = 1.05.
                    %    TP(6) - Zmniejszenie szybkości nauki,          default = 0.7.
                    %    TP(7) - Maksymalny współczynnik błędu,         default = 1.04.

                    TP = [disp_freq max_epoch err_goal lr lr_inc_vec(ind_lr_dec) lr_dec_vec(ind_lr_dec) er_vec(ind_er)];
                    [W1, B1, W2, B2, W3, B3, TE, TR] = trainbpa(W1, B1, 'tansig', W2, B2, 'tansig', W3, B3, 'purelin', P, T, TP);

                    %  SIMUFF will simulate networks with up to 3 layers.
                    %  
                    %  SIMUFF(P,W1,B1,'F1',...,Wn,Bn,'Fn')
                    %    P  - Matrix of input (column) vectors.
                    %    Wi - Weight matrix of the ith layer.
                    %    Bi - Bias (column) vector of the ith layer.
                    %    Fi - Transfer function of the ith layer (string).
                    %
                    %  Returns output of nth layer.

                    a = simuff(P, W1, B1, 'tansig', W2, B2, 'tansig', W3, B3, 'purelin');

                    q = (1 - sum(abs(T - a) > 0.5) / length(P)) * 100;

                    if q > qmax,
                        qmax = q;
                    end

                    ind_petli = ind_petli + 1;

                    [(ind_petli / liczba_petli * 100) q qmax]
                end
            end
        end
    end
end
