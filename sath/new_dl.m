% Set input and output directories
inputFolder = '/Users/praneethreddy/Downloads/USOD10k/USOD10k/USOD10K_TE/TE/RGB';
outputFolder = '/Users/praneethreddy/Downloads/Dl/output_te';

% Create output directory if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get list of all PNG images in the input directory
imageFiles = dir(fullfile(inputFolder, '*.png'));

% Loop through each image in the input folder
for i = 1:length(imageFiles)
    % Read image
    filename = imageFiles(i).name;
    filepath = fullfile(inputFolder, filename);
    rgbImage = double(imread(filepath)) / 255;

    % Apply underwater white balance and enhancement process
    % Load and split channels
    grayImage = rgb2gray(rgbImage);
    Ir = rgbImage(:,:,1);
    Ig = rgbImage(:,:,2);
    Ib = rgbImage(:,:,3);

    Ir_mean = mean(Ir, 'all');
    Ig_mean = mean(Ig, 'all');
    Ib_mean = mean(Ib, 'all');

    % Color compensation
    alpha = 0.1;
    Irc = Ir + alpha * (Ig_mean - Ir_mean);
    alpha = 0; 
    Ibc = Ib + alpha * (Ig_mean - Ib_mean);

    % White balance
    I = cat(3, Irc, Ig, Ibc);
    I_lin = rgb2lin(I);
    percentiles = 5;
    illuminant = illumgray(I_lin, percentiles);
    I_lin = chromadapt(I_lin, illuminant, 'ColorSpace', 'linear-rgb');
    Iwb = lin2rgb(I_lin);

    % Gamma correction
    Igamma = imadjust(Iwb, [], [], 2);

    % Image sharpening
    sigma = 20;
    Igauss = Iwb;
    N = 30;
    for iter = 1:N
        Igauss = imgaussfilt(Igauss, sigma);
        Igauss = min(Iwb, Igauss);
    end

    gain = 1;
    Norm = (Iwb - gain * Igauss);
    for n = 1:3
        Norm(:,:,n) = histeq(Norm(:,:,n));
    end
    Isharp = (Iwb + Norm) / 2;

    % Weights calculation
    Isharp_lab = rgb2lab(Isharp);
    Igamma_lab = rgb2lab(Igamma);

    % Image 1 weights
    R1 = double(Isharp_lab(:, :, 1)) / 255;
    WC1 = sqrt((((Isharp(:,:,1)) - (R1)).^2 + ((Isharp(:,:,2)) - (R1)).^2 + ((Isharp(:,:,3)) - (R1)).^2) / 3);
    WS1 = saliency_detection(Isharp);
    WS1 = WS1 / max(WS1, [], 'all');
    WSAT1 = sqrt(1 / 3 * ((Isharp(:,:,1) - R1).^2 + (Isharp(:,:,2) - R1).^2 + (Isharp(:,:,3) - R1).^2));

    % Image 2 weights
    R2 = double(Igamma_lab(:, :, 1)) / 255;
    WC2 = sqrt((((Igamma(:,:,1)) - (R2)).^2 + ((Igamma(:,:,2)) - (R2)).^2 + ((Igamma(:,:,3)) - (R2)).^2) / 3);
    WS2 = saliency_detection(Igamma);
    WS2 = WS2 / max(WS2, [], 'all');
    WSAT2 = sqrt(1 / 3 * ((Igamma(:,:,1) - R1).^2 + (Igamma(:,:,2) - R1).^2 + (Igamma(:,:,3) - R1).^2));

    % Normalized weights
    W1 = (WC1 + WS1 + WSAT1 + 0.1) ./ (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2 + 0.2);
    W2 = (WC2 + WS2 + WSAT2 + 0.1) ./ (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2 + 0.2);

    % Multi-scale fusion
    level = 10;
    Weight1 = gaussian_pyramid(W1, level);
    Weight2 = gaussian_pyramid(W2, level);

    R1 = laplacian_pyramid(Isharp(:, :, 1), level);
    G1 = laplacian_pyramid(Isharp(:, :, 2), level);
    B1 = laplacian_pyramid(Isharp(:, :, 3), level);

    R2 = laplacian_pyramid(Igamma(:, :, 1), level);
    G2 = laplacian_pyramid(Igamma(:, :, 2), level);
    B2 = laplacian_pyramid(Igamma(:, :, 3), level);

    % Fusion
    for k = 1:level
        Rr{k} = Weight1{k} .* R1{k} + Weight2{k} .* R2{k};
        Rg{k} = Weight1{k} .* G1{k} + Weight2{k} .* G2{k};
        Rb{k} = Weight1{k} .* B1{k} + Weight2{k} .* B2{k};
    end

    % Reconstruct the fused image
    R = pyramid_reconstruct(Rr);
    G = pyramid_reconstruct(Rg);
    B = pyramid_reconstruct(Rb);
    fusion = cat(3, R, G, B);

    % Save the output image
    outputFilePath = fullfile(outputFolder, filename);
    imwrite(fusion, outputFilePath);
end

disp('Processing complete. Check the output folder for enhanced images.');
