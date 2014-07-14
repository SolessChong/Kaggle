function Y = expandY(Yshr)
% 
% Expand Y from Y_shrink
%
% [1;3;2;5] 
% ==> 
% [1 0 0 0 0]
% [0 0 1 0 0]
% [0 1 0 0 0]
% [0 0 0 0 1]
%


Y = zeros(size(Yshr,1), 10);
for i = 1 : size(Y,1)
    Y(i, Yshr(i)+1) = 1;
end

