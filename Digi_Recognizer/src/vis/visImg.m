function visImg(imgMat, f)

global config

set(0, 'CurrentFigure', f);

n = size(imgMat, 1);
n1 = ceil(sqrt(n/2));
n2 = ceil(n/n1);

for k = 1 : n
    subplot(n1, n2, k);
    % reshape
    imagesc(reshape(imgMat(k,:), config.w, config.h)');
    axis image off
    colormap(1-gray);
end

end