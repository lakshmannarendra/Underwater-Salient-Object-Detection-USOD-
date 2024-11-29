% Set input and output directories
inputFolder = '/Users/praneethreddy/Downloads/USOD10k/USOD10k/USOD10K_TE/TE/RGB';
outputFolder = '/Users/praneethreddy/Downloads/Dl/output_te';

% Create output directory if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder); % Create the output directory
end

% Get list of all PNG images in the input directory
imageFiles = dir(fullfile(inputFolder, '*.png'));

% Loop through each image in the input folder
for i = 1:length(imageFiles)
    % Read image
    filename = imageFiles(i).name; % Get the image filename
    filepath = fullfile(inputFolder, filename); % Construct full file path
    rgbImage = double(imread(filepath)) / 255; % Read and normalize the image

    % Apply underwater white balance and enhancement process
    % Load and split channels
    grayImage = rgb2gray(rgbImage); % Convert RGB to grayscale
    Ir = rgbImage(:,:,1); % Red channel
    Ig = rgbImage(:,:,2); % Green channel
    Ib = rgbImage(:,:,3); % Blue channel

    Ir_mean = mean(Ir, 'all'); % Mean of red channel
    Ig_mean = mean(Ig, 'all'); % Mean of green channel
    Ib_mean = mean(Ib, 'all'); % Mean of blue channel

    % Color compensation
    alpha = 0.1; % Compensation factor for red channel
    Irc = Ir + alpha * (Ig_mean - Ir_mean); % Compensated red channel
    alpha = 0; % Compensation factor for blue channel
    Ibc = Ib + alpha * (Ig_mean - Ib_mean); % Compensated blue channel

    % White balance
    I = cat(3, Irc, Ig, Ibc); % Combine channels
    I_lin = rgb2lin(I); % Convert to linear RGB
    percentiles = 5; % Percentile for illuminant estimation
    illuminant = illumgray(I_lin, percentiles); % Estimate illuminant
    I_lin = chromadapt(I_lin, illuminant, 'ColorSpace', 'linear-rgb'); % Adapt colors
    Iwb = lin2rgb(I_lin); % Convert back to RGB

    % Gamma correction
    Igamma = imadjust(Iwb, [], [], 2); % Adjust gamma

    % Image sharpening
    sigma = 20; % Standard deviation for Gaussian filter
    Igauss = Iwb; % Initialize Gaussian image
    N = 30; % Number of iterations for sharpening
    for iter = 1:N
        Igauss = imgaussfilt(Igauss, sigma); % Apply Gaussian filter
        Igauss = min(Iwb, Igauss); % Limit to original image
    end

    gain = 1; % Gain factor for normalization
    Norm = (Iwb - gain * Igauss); % Calculate normalization
    for n = 1:3
        Norm(:,:,n) = histeq(Norm(:,:,n)); % Histogram equalization for each channel
    end
    Isharp = (Iwb + Norm) / 2; % Combine original and normalized images

    % Weights calculation
    Isharp_lab = rgb2lab(Isharp); % Convert to LAB color space
    Igamma_lab = rgb2lab(Igamma); % Convert gamma corrected image to LAB

    % Image 1 weights
    R1 = double(Isharp_lab(:, :, 1)) / 255; % L channel for sharp image
    WC1 = sqrt((((Isharp(:,:,1)) - (R1)).^2 + ((Isharp(:,:,2)) - (R1)).^2 + ((Isharp(:,:,3)) - (R1)).^2) / 3); % Weight calculation
    WS1 = saliency_detection(Isharp); % Saliency detection for sharp image
    WS1 = WS1 / max(WS1, [], 'all'); % Normalize saliency
    WSAT1 = sqrt(1 / 3 * ((Isharp(:,:,1) - R1).^2 + (Isharp(:,:,2) - R1).^2 + (Isharp(:,:,3) - R1).^2)); % Weight for sharp image

    % Image 2 weights
    R2 = double(Igamma_lab(:, :, 1)) / 255; % L channel for gamma corrected image
    WC2 = sqrt((((Igamma(:,:,1)) - (R2)).^2 + ((Igamma(:,:,2)) - (R2)).^2 + ((Igamma(:,:,3)) - (R2)).^2) / 3); % Weight calculation
    WS2 = saliency_detection(Igamma); % Saliency detection for gamma corrected image
    WS2 = WS2 / max(WS2, [], 'all'); % Normalize saliency
    WSAT2 = sqrt(1 / 3 * ((Igamma(:,:,1) - R1).^2 + (Igamma(:,:,2) - R1).^2 + (Igamma(:,:,3) - R1).^2)); % Weight for gamma corrected image

    % Normalized weights
    W1 = (WC1 + WS1 + WSAT1 + 0.1) ./ (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2 + 0.2); % Normalize weights for image 1
    W2 = (WC2 + WS2 + WSAT2 + 0.1) ./ (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2 + 0.2); % Normalize weights for image 2

    % Multi-scale fusion
    level = 10; % Number of levels for pyramid
    Weight1 = gaussian_pyramid(W1, level); % Gaussian pyramid for weights of image 1
    Weight2 = gaussian_pyramid(W2, level); % Gaussian pyramid for weights of image 2

    R1 = laplacian_pyramid(Isharp(:, :, 1), level); % Laplacian pyramid for red channel of image 1
    G1 = laplacian_pyramid(Isharp(:, :, 2), level); % Laplacian pyramid for green channel of image 1
    B1 = laplacian_pyramid(Isharp(:, :, 3), level); % Laplacian pyramid for blue channel of image 1

    R2 = laplacian_pyramid(Igamma(:, :, 1), level); % Laplacian pyramid for red channel of image 2
    G2 = laplacian_pyramid(Igamma(:, :, 2), level); % Laplacian pyramid for green channel of image 2
    B2 = laplacian_pyramid(Igamma(:, :, 3), level); % Laplacian pyramid for blue channel of image 2

    % Fusion
    for k = 1:level
        Rr{k} = Weight1{k} .* R1{k} + Weight2{k} .* R2{k}; % Fuse red channels
        Rg{k} = Weight1{k} .* G1{k} + Weight2{k} .* G2{k}; % Fuse green channels
        Rb{k} = Weight1{k} .* B1{k} + Weight2{k} .* B2{k}; % Fuse blue channels
    end

    % Reconstruct the fused image
    R = pyramid_reconstruct(Rr); % Reconstruct red channel
    G = pyramid_reconstruct(Rg); % Reconstruct green channel
    B = pyramid_reconstruct(Rb); % Reconstruct blue channel
    fusion = cat(3, R, G, B); % Combine channels into final image

    % Save the output image
    outputFilePath = fullfile(outputFolder, filename); % Construct output file path
    imwrite(fusion, outputFilePath); % Save the fused image
end

disp('Processing complete. Check the output folder for enhanced images.'); % Display completion message