function out = gaussian_pyramid(img, level)
% This function creates a Gaussian pyramid from the input image.
% 'img' is the input image, and 'level' is the number of pyramid levels.

h = 1/16 * [1, 4, 6, 4, 1]; % Create a Gaussian filter kernel
filt = h' * h; % Create a 2D Gaussian filter by multiplying the kernel with its transpose

out{1} = imfilter(img, filt, 'replicate', 'conv'); % Filter the original image and store it in the first level of the pyramid
temp_img = img; % Initialize a temporary image variable with the original image

for i = 2 : level
    % For each level from 2 to the specified number of levels:
    temp_img = temp_img(1 : 2 : end, 1 : 2 : end); % Downsample the image by taking every second pixel
    out{i} = imfilter(temp_img, filt, 'replicate', 'conv'); % Filter the downsampled image and store it in the pyramid
end