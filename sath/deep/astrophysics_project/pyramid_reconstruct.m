function out = pyramid_reconstruct(pyramid)
% This function reconstructs an image from a Gaussian pyramid.
% 'pyramid' is a cell array containing the images at different levels.

level = length(pyramid); % Get the number of levels in the pyramid
for i = level : -1 : 2
    % Loop from the top level of the pyramid down to the second level
    [m, n] = size(pyramid{i - 1}); % Get the size of the image at the current level
    %temp_pyramid = pyramid{i}; % (Commented out) This line would store the current level image in a temporary variable
    %out = pyramid{i - 1} + imresize(temp_pyramid, [m, n]); % (Commented out) This line would combine the current level with the previous level
    pyramid{i - 1} = pyramid{i - 1} + imresize(pyramid{i}, [m, n]); % Add the resized current level image to the previous level image
end
out = pyramid{1}; % The final output is the image at the first level of the pyramid