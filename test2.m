paperRoot = fileparts(mfilename('fullpath'));
clear;
clc;
close all;

%% 1. 读取数据
filePath = fullfile(paperRoot, 'mill', 'mill.mat');
raw = load(filePath);
mill = raw.mill;

nRecords = numel(mill);

%% 2. 初始化元数据
CaseID   = nan(nRecords,1);
RunID    = nan(nRecords,1);
VB       = nan(nRecords,1);
Time     = nan(nRecords,1);
DOC      = nan(nRecords,1);
Feed     = nan(nRecords,1);
Material = nan(nRecords,1);

signalNames = { ...
    "smcAC", ...
    "smcDC", ...
    "vib_table", ...
    "vib_spindle", ...
    "AE_table", ...
    "AE_spindle"};

nSignals = numel(signalNames);

signalLength = nan(nRecords,nSignals);
badSignal    = false(nRecords,nSignals);

%% 3. 逐条读取，避免空值造成程序报错
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

    if ~isempty(mill(i).time)
        Time(i) = mill(i).time;
    end

    if ~isempty(mill(i).DOC)
        DOC(i) = mill(i).DOC;
    end

    if ~isempty(mill(i).feed)
        Feed(i) = mill(i).feed;
    end

    if ~isempty(mill(i).material)
        Material(i) = mill(i).material;
    end

    for s = 1:nSignals
        x = mill(i).(signalNames{s});

        signalLength(i,s) = numel(x);

        if isempty(x) || any(~isfinite(x))
            badSignal(i,s) = true;
        end
    end
end

%% 4. 建立运行级元数据表
metadataTable = table( ...
    CaseID,RunID,VB,Time,DOC,Feed,Material, ...
    'VariableNames', ...
    {'CaseID','RunID','VB','Time','DOC','Feed','Material'});

%% 5. 输出总体审计结果
fprintf("========== 数据总体情况 ==========\n");
fprintf("总加工记录数：%d\n",nRecords);
fprintf("有效VB标签数：%d\n",sum(isfinite(VB)));
fprintf("缺失VB标签数：%d\n",sum(~isfinite(VB)));
fprintf("案例数量：%d\n",numel(unique(CaseID(isfinite(CaseID)))));
fprintf("材料编码：");
disp(unique(Material(isfinite(Material)))');

fprintf("DOC取值：");
disp(unique(DOC(isfinite(DOC)))');

fprintf("Feed取值：");
disp(unique(Feed(isfinite(Feed)))');

fprintf("\n各传感器信号长度范围：\n");

for s = 1:nSignals
    fprintf("%-15s 最短：%d，最长：%d，异常记录：%d\n", ...
        signalNames{s}, ...
        min(signalLength(:,s)), ...
        max(signalLength(:,s)), ...
        sum(badSignal(:,s)));
end

%% 6. 生成每个案例的汇总表
caseList = unique(CaseID(isfinite(CaseID)));

nCases = numel(caseList);

RecordCount = zeros(nCases,1);
ValidVBCount = zeros(nCases,1);
MissingVBCount = zeros(nCases,1);
VB_Min = nan(nCases,1);
VB_Max = nan(nCases,1);
CaseDOC = nan(nCases,1);
CaseFeed = nan(nCases,1);
CaseMaterial = nan(nCases,1);

for k = 1:nCases
    c = caseList(k);

    caseMask = CaseID == c;
    validMask = caseMask & isfinite(VB);

    RecordCount(k) = sum(caseMask);
    ValidVBCount(k) = sum(validMask);
    MissingVBCount(k) = sum(caseMask & ~isfinite(VB));

    if any(validMask)
        VB_Min(k) = min(VB(validMask));
        VB_Max(k) = max(VB(validMask));
    end

    firstIndex = find(caseMask,1,"first");

    CaseDOC(k) = DOC(firstIndex);
    CaseFeed(k) = Feed(firstIndex);
    CaseMaterial(k) = Material(firstIndex);
end

caseSummary = table( ...
    caseList,RecordCount,ValidVBCount,MissingVBCount, ...
    VB_Min,VB_Max,CaseDOC,CaseFeed,CaseMaterial, ...
    'VariableNames', ...
    {'CaseID','Records','ValidVB','MissingVB', ...
     'VB_Min','VB_Max','DOC','Feed','Material'});

fprintf("\n========== 每个案例汇总 ==========\n");
disp(caseSummary);

%% 7. 检查CaseID和RunID组合是否重复
runKey = string(CaseID) + "_" + string(RunID);

fprintf("重复的CaseID-RunID组合数量：%d\n", ...
    numel(runKey)-numel(unique(runKey)));

%% 8. 绘制每个案例的磨损变化
figure("Color","w","Position",[100 50 1200 850]);
tiledlayout(4,4,"TileSpacing","compact","Padding","compact");

for k = 1:nCases
    c = caseList(k);

    mask = CaseID == c & isfinite(VB);

    currentRun = RunID(mask);
    currentVB = VB(mask);

    [currentRun,order] = sort(currentRun);
    currentVB = currentVB(order);

    nexttile;
    plot(currentRun,currentVB,"o-", ...
        "LineWidth",1.1,"MarkerSize",4);

    grid on;
    xlabel("Run");
    ylabel("VB");
    title("Case "+string(c));
end

sgtitle("Tool Wear Progression for All Cases");

%% 9. 绘制VB总体分布
figure("Color","w");

histogram(VB(isfinite(VB)),15);

grid on;
xlabel("Flank wear VB");
ylabel("Number of runs");
title("Distribution of Valid VB Labels");

%% 10. 保存审计结果
outputFolder = fullfile(paperRoot, 'mill', 'results');

if ~exist(outputFolder,"dir")
    mkdir(outputFolder);
end

writetable(metadataTable, ...
    fullfile(outputFolder,"metadata_table.csv"));

writetable(caseSummary, ...
    fullfile(outputFolder,"case_summary.csv"));

save(fullfile(outputFolder,"audit_results.mat"), ...
    "metadataTable","caseSummary", ...
    "signalLength","badSignal");