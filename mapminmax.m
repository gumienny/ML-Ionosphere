function [N] = mapminmax(P, ymin, ymax)
    xmax = max(P');
    xmin = min(P');

    N = zeros(size(P));

    for i = 1:length(xmax)
        N(i, :) = (ymax - ymin) * (P(i, :) - xmin(i)) / (xmax(i) - xmin(i)) + ymin;
    endfor
endfunction
