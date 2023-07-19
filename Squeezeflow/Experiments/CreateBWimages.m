function [BW,level] = createBWimages(image)
% binarize image
level = graythresh(image);
BW = imbinarize(image,level);
BW = bwareaopen(BW,2000,8);

% make negative image
BW = abs(BW - 1);
BW = bwareaopen(BW,50,8);
end