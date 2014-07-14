function Y = shrinkY(Yexp)
%
% Shrink Y from Y_expand
%
% [1 0 0 0 0]
% [0 0 1 0 0]
% [0 1 0 0 0]
% [0 0 0 0 1]
% ==> 
% [1;3;2;5] 
%

Y = max(Yexp~=0, [], 2);