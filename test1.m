paperRoot = fileparts(mfilename('fullpath'));
clear;
clc;
close all;

filePath = fullfile(paperRoot, 'mill', 'mill.mat');
raw = load(filePath);
mill = raw.mill;

fprintf("记录数量：%d\n",numel(mill));

fieldNames = fieldnames(mill);

fprintf("\n结构体字段：\n");
disp(fieldNames);

fprintf("\n第一条记录各字段的类型和大小：\n");

for k = 1:numel(fieldNames)
    fieldValue = mill(1).(fieldNames{k});

    fprintf("%-15s 类型：%-10s 大小：%s\n", ...
        fieldNames{k},class(fieldValue),mat2str(size(fieldValue)));
end