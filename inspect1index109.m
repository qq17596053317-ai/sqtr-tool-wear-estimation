paperRoot = fileparts(mfilename('fullpath'));
clear;
clc;
close all;

%% 1. 路径和数据
filePath = ...
    fullfile(paperRoot, 'mill', 'mill.mat');

resultFolder = ...
    fullfile(paperRoot, 'mill', 'results');

raw = load(filePath);
mill = raw.mill;

featureData = load( ...
    fullfile(resultFolder, ...
    "run_level_features.mat"));

featureTable = featureData.featureTable;

Fs = 250;
targetIndex = 109;

%% 2. 读取目标信号
x = double(mill(targetIndex).smcDC(:));
n = numel(x);

startIndex = floor(0.25*n)+1;
endIndex = floor(0.75*n);

xCentral = x(startIndex:endIndex);

timeFull = (0:n-1)/Fs;
timeCentral = ...
    (startIndex-1:endIndex-1)/Fs;

%% 3. 鲁棒异常点统计
centralMedian = median(xCentral);
centralMAD = mad(xCentral,1);

if centralMAD < eps
    robustZ = zeros(size(xCentral));
else
    robustZ = ...
        0.6745*(xCentral-centralMedian)/centralMAD;
end

extremeMask5 = abs(robustZ) > 5;
extremeMask10 = abs(robustZ) > 10;

fprintf("========== OriginalIndex 109原始信号统计 ==========\n");
fprintf("Case：%d\n",mill(targetIndex).case);
fprintf("Run：%d\n",mill(targetIndex).run);
fprintf("VB：%.4f\n",mill(targetIndex).VB);
fprintf("信号长度：%d\n",n);
fprintf("中心区域长度：%d\n",numel(xCentral));
fprintf("中心区域最小值：%.6f\n",min(xCentral));
fprintf("中心区域最大值：%.6f\n",max(xCentral));
fprintf("中心区域均值：%.6f\n",mean(xCentral));
fprintf("中心区域中位数：%.6f\n",centralMedian);
fprintf("中心区域标准差：%.6f\n",std(xCentral));
fprintf("中心区域MAD：%.6f\n",centralMAD);
fprintf("|Robust Z| > 5的点数：%d\n",sum(extremeMask5));
fprintf("|Robust Z| > 10的点数：%d\n",sum(extremeMask10));

%% 4. 绘制目标信号
figure( ...
    "Color","w", ...
    "Position",[100 80 1300 850]);

tiledlayout(2,2, ...
    "TileSpacing","compact", ...
    "Padding","compact");

% 完整信号
nexttile;
plot(timeFull,x,"b","LineWidth",0.8);
hold on;
xline((startIndex-1)/Fs,"r--","Central 50% start");
xline((endIndex-1)/Fs,"r--","Central 50% end");
hold off;
grid on;
xlabel("Time (s)");
ylabel("smcDC");
title("OriginalIndex 109 — Full signal");

% 中心50%
nexttile;
plot(timeCentral,xCentral,"k","LineWidth",0.8);
hold on;

if any(extremeMask5)
    scatter( ...
        timeCentral(extremeMask5), ...
        xCentral(extremeMask5), ...
        25,"r","filled");
end

hold off;
grid on;
xlabel("Time (s)");
ylabel("smcDC");
title("Central 50% with robust outliers");
legend("Signal","|Robust Z| > 5");

% 直方图
nexttile;
histogram(xCentral,50);
grid on;
xlabel("smcDC");
ylabel("Count");
title("Central-region distribution");

% 鲁棒Z分数
nexttile;
plot(timeCentral,robustZ,"LineWidth",0.8);
hold on;
yline(5,"r--");
yline(-5,"r--");
hold off;
grid on;
xlabel("Time (s)");
ylabel("Robust Z-score");
title("Robust Z-score of central signal");

sgtitle( ...
    "Inspection of Anomalous smcDC Signal — Index 109");

%% 5. 比较Case 12最后几次有效加工
caseMask = ...
    featureTable.CaseID == 12;

caseRows = find(caseMask);

[~,order] = sort( ...
    featureTable.RunID(caseRows));

caseRows = caseRows(order);

% 最后最多4次有效加工
numberToPlot = min(4,numel(caseRows));
selectedRows = ...
    caseRows(end-numberToPlot+1:end);

figure( ...
    "Color","w", ...
    "Position",[100 100 1300 600]);

tiledlayout(1,2, ...
    "TileSpacing","compact", ...
    "Padding","compact");

%% 归一化时间下的中心信号比较
nexttile;
hold on;

legendText = strings(numberToPlot,1);

for k = 1:numberToPlot

    sourceIndex = ...
        featureTable.OriginalIndex( ...
        selectedRows(k));

    currentSignal = ...
        double(mill(sourceIndex).smcDC(:));

    currentN = numel(currentSignal);

    currentStart = ...
        floor(0.25*currentN)+1;

    currentEnd = ...
        floor(0.75*currentN);

    currentCentral = ...
        currentSignal(currentStart:currentEnd);

    normalizedTime = linspace(0,1, ...
        numel(currentCentral));

    plot( ...
        normalizedTime, ...
        currentCentral, ...
        "LineWidth",0.9);

    legendText(k) = ...
        "Run "+ ...
        string(featureTable.RunID(selectedRows(k)))+ ...
        ", VB="+ ...
        string(featureTable.VB(selectedRows(k)));
end

hold off;
grid on;
xlabel("Normalized central-region time");
ylabel("smcDC");
title("Last valid runs of Case 12");
legend(legendText,"Location","best");

%% 三个异常敏感特征随Run变化
nexttile;

yyaxis left;

plot( ...
    featureTable.RunID(caseRows), ...
    featureTable.smcDC_Kurtosis(caseRows), ...
    "o-","LineWidth",1.2);

ylabel("Kurtosis");

yyaxis right;

plot( ...
    featureTable.RunID(caseRows), ...
    featureTable.smcDC_CrestFactor(caseRows), ...
    "s-","LineWidth",1.2);

ylabel("Crest factor");

grid on;
xlabel("Run");
title("Case 12 smcDC feature progression");
legend("Kurtosis","Crest factor", ...
    "Location","best");

%% 6. 保存图片
figureHandles = findall(groot,"Type","figure");

for k = 1:numel(figureHandles)

    outputName = sprintf( ...
        "index109_inspection_figure_%d.png", ...
        figureHandles(k).Number);

    exportgraphics( ...
        figureHandles(k), ...
        fullfile(resultFolder,outputName), ...
        "Resolution",300);
end