paperRoot = fileparts(mfilename('fullpath'));
clear;
clc;
close all;

%% 1. 读取数据
filePath = fullfile(paperRoot, 'mill', 'mill.mat');
raw = load(filePath);
mill = raw.mill;

nRecords = numel(mill);

%% 2. 提取CaseID、RunID和VB
CaseID = nan(nRecords,1);
RunID  = nan(nRecords,1);
VB     = nan(nRecords,1);

for i = 1:nRecords
    if ~isempty(mill(i).case)
        CaseID(i) = mill(i).case;
    end

    if ~isempty(mill(i).run)
        RunID(i) = mill(i).run;
    end

    if ~isempty(mill(i).VB)
        VB(i) = mill(i).VB;
    end
end

%% 3. 选择Case 11
caseToStudy = 11;

caseIndices = find( ...
    CaseID == caseToStudy & isfinite(VB));

caseVB = VB(caseIndices);

% 按VB从小到大排序
[sortedVB,order] = sort(caseVB);
sortedIndices = caseIndices(order);

% 选择最小、中位和最大磨损
nValid = numel(sortedIndices);

selectedPositions = [ ...
    1, ...
    round(nValid/2), ...
    nValid];

selectedIndices = sortedIndices(selectedPositions);

selectedRuns = RunID(selectedIndices);
selectedVB = VB(selectedIndices);

selectedTable = table( ...
    selectedIndices,selectedRuns,selectedVB, ...
    'VariableNames', ...
    {'StructureIndex','RunID','VB'});

fprintf("选择的轻、中、重磨损记录：\n");
disp(selectedTable);

%% 4. 信号名称及绘图参数
signalNames = { ...
    "smcAC", ...
    "smcDC", ...
    "vib_table", ...
    "vib_spindle", ...
    "AE_table", ...
    "AE_spindle"};

Fs = 250;

colors = lines(3);

stageNames = { ...
    "Light wear", ...
    "Medium wear", ...
    "Heavy wear"};

%% 5. 绘制时域和功率谱
figure( ...
    "Color","w", ...
    "Position",[50 30 1500 1000]);

tiledlayout(6,2, ...
    "TileSpacing","compact", ...
    "Padding","compact");

for s = 1:numel(signalNames)

    %% 左侧：时域信号
    nexttile;
    hold on;

    for k = 1:numel(selectedIndices)
        x = mill(selectedIndices(k)).(signalNames{s});
        x = x(:);

        timeAxis = (0:numel(x)-1)/Fs;

        plot( ...
            timeAxis,x, ...
            "Color",colors(k,:), ...
            "LineWidth",0.7);
    end

    hold off;
    grid on;

    xlabel("Time (s)");
    ylabel("Amplitude");
    title(signalNames{s}+" — Time domain", ...
        "Interpreter","none");

    legendText = strings(3,1);

    for k = 1:3
        legendText(k) = ...
            stageNames{k}+ ...
            ", Run="+string(selectedRuns(k))+ ...
            ", VB="+string(selectedVB(k));
    end

    legend(legendText,"Location","best");

    %% 右侧：Welch功率谱
    nexttile;
    hold on;

    for k = 1:numel(selectedIndices)
        x = mill(selectedIndices(k)).(signalNames{s});
        x = x(:);

        % 去除直流趋势，避免均值主导频谱
        xDetrended = detrend(x);

        window = hamming(512);
        overlap = 256;
        nfft = 1024;

        [Pxx,f] = pwelch( ...
            xDetrended, ...
            window, ...
            overlap, ...
            nfft, ...
            Fs);

        plot( ...
            f,10*log10(Pxx+eps), ...
            "Color",colors(k,:), ...
            "LineWidth",1);
    end

    hold off;
    grid on;

    xlabel("Frequency (Hz)");
    ylabel("PSD (dB/Hz)");
    title(signalNames{s}+" — Power spectrum", ...
        "Interpreter","none");
end

sgtitle( ...
    "NASA Milling Dataset — Case 11 Wear-Stage Comparison");

%% 6. 保存图片
outputFolder = ...
    fullfile(paperRoot, 'mill', 'results');

if ~exist(outputFolder,"dir")
    mkdir(outputFolder);
end

exportgraphics( ...
    gcf, ...
    fullfile(outputFolder, ...
    "case11_wear_stage_comparison.png"), ...
    "Resolution",300);