paperRoot = fileparts(mfilename('fullpath'));
clear;
clc;
close all;

%% 1. 读取原始数据
filePath = fullfile(paperRoot, 'mill', 'mill.mat');
outputFolder = fullfile(paperRoot, 'mill', 'results');

if ~exist(outputFolder,"dir")
    mkdir(outputFolder);
end

raw = load(filePath);
mill = raw.mill;

nRecords = numel(mill);

%% 2. 信号和特征名称
signalNames = [ ...
    "smcAC", ...
    "smcDC", ...
    "vib_table", ...
    "vib_spindle", ...
    "AE_table", ...
    "AE_spindle"];

baseFeatureNames = [ ...
    "Mean", ...
    "RMS", ...
    "Std", ...
    "PeakToPeak", ...
    "Skewness", ...
    "Kurtosis", ...
    "CrestFactor", ...
    "MeanFrequency", ...
    "MedianFrequency", ...
    "SpectralEntropy", ...
    "LowBandRatio", ...
    "MidBandRatio", ...
    "HighBandRatio"];

Fs = 250;

nSignals = numel(signalNames);
nFeaturesPerSignal = numel(baseFeatureNames);
nTotalFeatures = nSignals*nFeaturesPerSignal;

%% 3. 找到具有有效VB的记录
validRecord = false(nRecords,1);

for i = 1:nRecords
    currentVB = mill(i).VB;

    validRecord(i) = ...
        ~isempty(currentVB) && ...
        isnumeric(currentVB) && ...
        isscalar(currentVB) && ...
        isfinite(currentVB);
end

validIndices = find(validRecord);
nValid = numel(validIndices);

fprintf("有效加工记录数量：%d\n",nValid);

%% 4. 初始化元数据和特征矩阵
OriginalIndex = validIndices;
CaseID = nan(nValid,1);
RunID = nan(nValid,1);
VB = nan(nValid,1);
DOC = nan(nValid,1);
Feed = nan(nValid,1);
Material = nan(nValid,1);

X = nan(nValid,nTotalFeatures);

featureNames = strings(1,nTotalFeatures);

columnStart = 1;

for s = 1:nSignals
    columnEnd = columnStart+nFeaturesPerSignal-1;

    featureNames(columnStart:columnEnd) = ...
        signalNames(s)+"_"+baseFeatureNames;

    columnStart = columnEnd+1;
end

%% 5. 逐条提取特征
for row = 1:nValid

    sourceIndex = validIndices(row);

    CaseID(row) = mill(sourceIndex).case;
    RunID(row) = mill(sourceIndex).run;
    VB(row) = mill(sourceIndex).VB;
    DOC(row) = mill(sourceIndex).DOC;
    Feed(row) = mill(sourceIndex).feed;
    Material(row) = mill(sourceIndex).material;

    columnStart = 1;

    for s = 1:nSignals

        x = mill(sourceIndex).(signalNames(s));
        x = double(x(:));

        %% 选择中间50%区域
        n = numel(x);

        startIndex = floor(0.25*n)+1;
        endIndex = floor(0.75*n);

        xSelected = x(startIndex:endIndex);

        %% 提取特征
        currentFeatures = calculateSignalFeatures( ...
            xSelected,Fs);

        columnEnd = columnStart+nFeaturesPerSignal-1;

        X(row,columnStart:columnEnd) = currentFeatures;

        columnStart = columnEnd+1;
    end
end

%% 6. 建立特征表
metadataTable = table( ...
    OriginalIndex,CaseID,RunID,VB,DOC,Feed,Material);

featureTableOnly = array2table( ...
    X, ...
    "VariableNames",cellstr(featureNames));

featureTable = [metadataTable featureTableOnly];

%% 7. 检查结果
fprintf("\n========== 特征表检查 ==========\n");
fprintf("特征表行数：%d\n",height(featureTable));
fprintf("特征表总列数：%d\n",width(featureTable));
fprintf("传感器特征列数：%d\n",nTotalFeatures);
fprintf("非有限特征数量：%d\n",sum(~isfinite(X),"all"));

fprintf("\n前5行元数据：\n");
disp(featureTable(1:5,1:7));

fprintf("\n前10个特征名称：\n");
disp(featureNames(1:10)');

%% 8. 检查各特征的变化范围
featureMinimum = min(X,[],1);
featureMaximum = max(X,[],1);
featureStandardDeviation = std(X,[],1);

featureSummary = table( ...
    featureNames', ...
    featureMinimum', ...
    featureMaximum', ...
    featureStandardDeviation', ...
    'VariableNames', ...
    {'Feature','Minimum','Maximum','StandardDeviation'});

fprintf("\n前15个特征的统计结果：\n");
disp(featureSummary(1:15,:));

%% 9. 保存结果
save( ...
    fullfile(outputFolder,"run_level_features.mat"), ...
    "featureTable","featureSummary","Fs");

writetable( ...
    featureTable, ...
    fullfile(outputFolder,"run_level_features.csv"));

writetable( ...
    featureSummary, ...
    fullfile(outputFolder,"feature_summary.csv"));

fprintf("\n特征文件已保存到：\n%s\n",outputFolder);

%% 局部函数：计算单路信号特征
function features = calculateSignalFeatures(x,Fs)

    x = double(x(:));

    %% 原始信号特征
    featureMean = mean(x);
    featureRMS = rms(x);
    featureStd = std(x);
    featurePeakToPeak = max(x)-min(x);

    %% 去趋势信号
    xDetrended = detrend(x);

    featureSkewness = skewness(xDetrended);
    featureKurtosis = kurtosis(xDetrended);

    detrendedRMS = rms(xDetrended);

    featureCrestFactor = ...
        max(abs(xDetrended))/max(detrendedRMS,eps);

    %% Welch功率谱
    windowLength = min(512,numel(xDetrended));

    analysisWindow = hamming(windowLength);

    overlapLength = floor(windowLength/2);

    nfft = max(1024,2^nextpow2(windowLength));

    [Pxx,f] = pwelch( ...
        xDetrended, ...
        analysisWindow, ...
        overlapLength, ...
        nfft, ...
        Fs);

    totalPower = sum(Pxx);

    if totalPower <= eps
        normalizedPower = ...
            ones(size(Pxx))/numel(Pxx);
    else
        normalizedPower = Pxx/totalPower;
    end

    %% 平均频率
    featureMeanFrequency = ...
        sum(f.*normalizedPower);

    %% 中值频率
    cumulativePower = cumsum(normalizedPower);

    medianIndex = find( ...
        cumulativePower >= 0.5, ...
        1, ...
        "first");

    featureMedianFrequency = f(medianIndex);

    %% 归一化频谱熵
    featureSpectralEntropy = ...
        -sum(normalizedPower.*log2(normalizedPower+eps)) / ...
        log2(numel(normalizedPower));

    %% 相对频带能量
    lowMask = f >= 0 & f < 20;
    midMask = f >= 20 & f < 60;
    highMask = f >= 60 & f <= Fs/2;

    featureLowBandRatio = ...
        sum(normalizedPower(lowMask));

    featureMidBandRatio = ...
        sum(normalizedPower(midMask));

    featureHighBandRatio = ...
        sum(normalizedPower(highMask));

    %% 输出顺序必须与baseFeatureNames一致
    features = [ ...
        featureMean, ...
        featureRMS, ...
        featureStd, ...
        featurePeakToPeak, ...
        featureSkewness, ...
        featureKurtosis, ...
        featureCrestFactor, ...
        featureMeanFrequency, ...
        featureMedianFrequency, ...
        featureSpectralEntropy, ...
        featureLowBandRatio, ...
        featureMidBandRatio, ...
        featureHighBandRatio];
end