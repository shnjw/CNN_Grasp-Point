clear

imageFolder = 'data'; % 이미지 폴더 경로
txtFolder = 'data'; % 좌표 txt 파일이 저장된 폴더

% 이미지 데이터스토어 생성
imds = imageDatastore(imageFolder, 'FileExtensions', '.png');

% 레이블 추가 (모든 이미지를 같은 레이블로 설정하여 splitEachLabel 함수 사용)
numImages = numel(imds.Files);
imds.Labels = ones(numImages, 1); % 모든 이미지에 동일한 레이블 추가

% 데이터 분할 비율 설정 
trainingSplitRatio = 0.8;
[imdsTrain, imdsValidation] = splitEachLabel(imds, trainingSplitRatio, 'randomized');

% 이미지에 대한 좌표 읽기 함수 정의 (파일 이름에서 r을 제거하고 cpos를 붙임)
% 이미지 크기를 224x224로 리사이즈할 때 좌표도 변환
function points = readPointsFromTxt(imageFile, txtFolder)
    % 이미지 파일 이름에서 확장자를 제거하고, 맨 뒤의 'r'을 제거한 후 'cpos'를 붙임
    [~, name, ~] = fileparts(imageFile);
    if name(end) == 'r'
        name = name(1:end-1); % 마지막 문자가 'r'이면 제거
    end
    txtFile = fullfile(txtFolder, [name, 'cpos.txt']); % 'cpos' 추가

    % txt 파일에서 8개의 점 좌표 읽기 (각 줄에 x, y 좌표가 있다고 가정)
    fileID = fopen(txtFile, 'r');
    coords = fscanf(fileID, '%f %f', [2, 8]);
    fclose(fileID);
    
    % 좌표를 224x224 이미지에 맞게 스케일링 (640x480 기준)
    coords(1, :) = coords(1, :) * (224 / 640); % x좌표 변환
    coords(2, :) = coords(2, :) * (224 / 480); % y좌표 변환
    
    % 변환된 점 좌표 반환
    points = coords(:)'; % [x1, y1, x2, y2, ..., x8, y8] 형식으로 반환
end

numTrainImages = numel(imdsTrain.Files);
numValidationImages = numel(imdsValidation.Files);
trainCoordinates = zeros(numTrainImages, 16); % 8개의 점 좌표를 저장할 배열
validationCoordinates = zeros(numValidationImages, 16); % 8개의 점 좌표를 저장할 배열
%%
for i = 1:numTrainImages
    imageFile = imdsTrain.Files{i};
    trainCoordinates(i, :) = readPointsFromTxt(imageFile, txtFolder);
end
for i = 1:numValidationImages
    imageFile = imdsValidation.Files{i};
    validationCoordinates(i, :) = readPointsFromTxt(imageFile, txtFolder);
end

% 훈련용 이미지와 좌표 데이터를 테이블로 결합
trainTbl = table(imdsTrain.Files, trainCoordinates, 'VariableNames', {'imageFile', 'points'});
% 검증용 이미지와 좌표 데이터를 테이블로 결합
validationTbl = table(imdsValidation.Files, validationCoordinates, 'VariableNames', {'imageFile', 'points'});
% 훈련용 데이터스토어 생성
dsTrain = fileDatastore(trainTbl.imageFile, 'ReadFcn', @(x)imresize(imread(x), [224 224]), 'FileExtensions', '.png');
responseDSTrain = arrayDatastore(trainTbl.points);
% 검증용 데이터스토어 생성
dsValidation = fileDatastore(validationTbl.imageFile, 'ReadFcn', @(x)imresize(imread(x), [224 224]), 'FileExtensions', '.png');
responseDSValidation = arrayDatastore(validationTbl.points);
% 훈련 데이터와 레이블 결합
combinedDSTrain = combine(dsTrain, responseDSTrain);
% 검증 데이터와 레이블 결합
combinedDSValidation = combine(dsValidation, responseDSValidation);
%%

% 회귀 네트워크 구성 (8개의 (x, y) 좌표 예측)
layers = [
    imageInputLayer([224 224 3], 'Name', 'input')

    % 1st Convolution Block
    convolution2dLayer(3, 64, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'batchnorm_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

    % 2nd Convolution Block
    convolution2dLayer(3, 128, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'batchnorm_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

    % 3rd Convolution Block
    convolution2dLayer(3, 256, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'batchnorm_3')
    reluLayer('Name', 'relu_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')

    % 4th Convolution Block 
    convolution2dLayer(3, 512, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_4')
    batchNormalizationLayer('Name', 'batchnorm_4')
    reluLayer('Name', 'relu_4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4')

    % 5th Convolution Block 
    convolution2dLayer(3, 512, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_5')
    batchNormalizationLayer('Name', 'batchnorm_5')
    reluLayer('Name', 'relu_5')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool5')

    % 6th Convolution Block
    convolution2dLayer(3, 1024, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_6')
    batchNormalizationLayer('Name', 'batchnorm_6')
    reluLayer('Name', 'relu_6')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool6')

    % Dropout Layer to prevent overfitting
    dropoutLayer(0.4, 'Name', 'dropout')

    % Fully connected layers 
    fullyConnectedLayer(2048, 'Name', 'fc_1')
    reluLayer('Name', 'relu_fc_1')

    fullyConnectedLayer(1024, 'Name', 'fc_2')
    reluLayer('Name', 'relu_fc_2')

    fullyConnectedLayer(512, 'Name', 'fc_3')
    reluLayer('Name', 'relu_fc_3')

    fullyConnectedLayer(16, 'Name', 'fc_output') % 최종 출력 (8개의 (x, y) 좌표)
    regressionLayer('Name', 'regression_output')
];

% 훈련 옵션 설정 
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 50, ... 
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', combinedDSValidation, ... 
    'ValidationFrequency', 60, ... 
    'Verbose', true, ...
    'Plots', 'training-progress');

% 네트워크 훈련 
net = trainNetwork(combinedDSTrain, layers, options);
%%
% 네트워크 저장
save("GraspPoint5", "net")