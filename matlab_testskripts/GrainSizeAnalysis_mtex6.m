function [EBSDdata_smDat, GSheader] = GrainSizeAnalysis_mtex6( ...
        grains_in, datname, filename, particleRem, minLIL, step, ...
        foldername, n, ebsd1plot, CS, ebsd_smoothedFill)

% legacy args not used:
% grains_in, filename, minLIL

GSheader = { ...
    'Avg. Max. Length X',   '+- Max. Length X', ...
    'Avg. Max. Length Y',   '+- Max. Length Y', ...
    'Avg. Equiv. Diameter', '+- Equiv. Diameter', ...
    'Avg. Aspect Ratio',    '+- Aspect Ratio', ...
    'Area-weighted ECD'};

EBSDdata_smDat = nan(1,9);

%% 1) Phase name
if particleRem == 1 && iscell(CS) && numel(CS) > 1
    ph = CS{2}.mineral;
else
    ph = 'indexed';
end

%% 2) Preflight
if ~isfinite(step) || step <= 0
    error('[%s] step must be > 0 (µm). Got step = %g', char(datname), step);
end

% get EBSD phase (must exist)
try
    ebsd_ph = ebsd_smoothedFill(ph);
catch
    error('[%s] ebsd_smoothedFill does not contain phase "%s".', char(datname), char(ph));
end

if numel(ebsd_ph) == 0
    warning('[%s] ebsd_smoothedFill(%s) is empty.', char(datname), char(ph));
    return;
end

% ensure grainId exists by reconstructing here (robust)
% choose your grain angle; keep it consistent with your pipeline
ang = 5*degree;
[grains, ebsd_ph.grainId] = calcGrains(ebsd_ph, 'angle', ang, 'minPixel', 3);  % minPixel optional [web:26]

% keep only phase grains (calcGrains returns all phases present in ebsd_ph)
ALgrains = grains(ph);

fprintf('[%s] reconstructed grains(%s): %d\n', char(datname), char(ph), length(ALgrains));

if isempty(ALgrains)
    warning('[%s] No grains after reconstruction for phase %s.', char(datname), char(ph));
    return;
end

% grainSize sanity: number of pixels per grain [web:77][web:69]
gs = double(ALgrains.grainSize(:));
fprintf('[%s] grainSize: min=%g median=%g max=%g nnz=%d/%d\n', ...
    char(datname), min(gs), median(gs), max(gs), nnz(gs), numel(gs));

if nnz(gs) == 0
    warning('[%s] All grainSize are zero even after calcGrains -> cannot proceed.', char(datname));
    return;
end

%% 3) Boundary-grain removal via EBSD edge pixels (on reconstructed grainId)
% edge pixels defined on the same ebsd_ph used for calcGrains
inset = 1.0 * step;
xmin = min(ebsd_ph.x) + inset;  xmax = max(ebsd_ph.x) - inset;
ymin = min(ebsd_ph.y) + inset;  ymax = max(ebsd_ph.y) - inset;

edgePts = ebsd_ph.x <= xmin | ebsd_ph.x >= xmax | ...
          ebsd_ph.y <= ymin | ebsd_ph.y >= ymax;

edgeIds = unique(ebsd_ph.grainId(edgePts));
edgeIds(edgeIds <= 0) = [];

keepIds = ALgrains.id(~ismember(ALgrains.id, edgeIds));
ALgrains_int = ALgrains('id', keepIds);

% safeguard: if too aggressive, keep all
if length(ALgrains_int) < 50 || length(ALgrains_int)/length(ALgrains) < 0.15
    warning('[%s] boundary removal too aggressive -> keeping all grains.', char(datname));
else
    ALgrains = ALgrains_int;
end

fprintf('[%s] grains after boundary step: %d\n', char(datname), length(ALgrains));

%% 4) ECD from pixel area (avoids polySgnArea3 entirely)
Apx = double(ALgrains.grainSize) * (double(step)^2);
ECD = 2 * sqrt(Apx / pi);

valid = isfinite(ECD) & isreal(ECD) & ECD > 0 & ECD < 900;
ALgrains = ALgrains('id', ALgrains.id(valid));
ECD      = ECD(valid);

if isempty(ALgrains)
    warning('No valid grains after ECD filtering for %s.', char(datname));
    return;
end

%% 5) AR (guarded)
try
    AR = double(ALgrains.aspectRatio(:));
catch
    AR = nan(size(ECD));
end

%% 6) Plot ECD map (optional)
try
    figtemp = figure;
    plot(ebsd1plot, ebsd1plot.bc, 'figSize', 'large');
    colormap gray; setColorRange([0 220]);
    hold on; freezeColors;
    plot(ALgrains, ECD, 'FaceAlpha', 0.7);
    colormap parula; setColorRange([0 140]);
    legend('off');
    mtexTitle('Grain Size (ECD in $\mu$m)');
    mtexColorbar;
    saveFigure(['GrainECD_' char(datname) '.png']);
    close(figtemp);
catch
end

%% 7) MaxDimX/Y from vertices (guarded)
nG = numel(ALgrains);
maxDimX = nan(nG,1);
maxDimY = nan(nG,1);
for i = 1:nG
    try
        vi = ALgrains(i).V;
        maxDimX(i) = max(vi.x) - min(vi.x);
        maxDimY(i) = max(vi.y) - min(vi.y);
    catch
    end
end
mDim = isfinite(maxDimX) & isfinite(maxDimY) & maxDimX>0 & maxDimY>0;
maxDimX = maxDimX(mDim);
maxDimY = maxDimY(mDim);

avg_maxDimX = mean(maxDimX,'omitnan'); std_maxDimX = std(maxDimX,'omitnan');
avg_maxDimY = mean(maxDimY,'omitnan'); std_maxDimY = std(maxDimY,'omitnan');

%% 8) Export histograms to Excel (same layout)
binWidth   = 5;
binEdges   = 0:binWidth:140;
binCenters = binEdges(1:end-1) + binWidth/2;
nBins      = numel(binCenters);

countsX = histcounts(maxDimX, binEdges, 'Normalization','probability');
countsY = histcounts(maxDimY, binEdges, 'Normalization','probability');

docFile = fullfile(foldername, 'Documentation.xlsx');
sheetGS = 'Grain Size hist';
colLet  = excelColumn(n + 1);

writecell({char(datname)}, docFile, 'Sheet', sheetGS, 'Range', [colLet '1']);

blocks = { ...
    2,           binCenters(:), countsX(:), 'maxDimX'; ...
    nBins + 3,   binCenters(:), countsY(:), 'maxDimY'; ...
    2*nBins + 4, binCenters(:), [],         'area weighted ECD'};

for b = 1:size(blocks, 1)
    rLabel = blocks{b,1};
    vals   = blocks{b,3};
    writecell({'BinCenters', blocks{b,4}}, docFile, 'Sheet', sheetGS, 'Range', ['A' num2str(rLabel)]);
    writematrix(blocks{b,2}, docFile, 'Sheet', sheetGS, 'Range', ['A' num2str(rLabel + 1)]);
    if ~isempty(vals)
        writematrix(vals, docFile, 'Sheet', sheetGS, 'Range', [colLet num2str(rLabel + 1)]);
    end
end

%% 9) Area-weighted ECD (pixel-area weights = Apx)
w = Apx(valid);                         % same subset as ECD/ALgrains
meanAreaWeightedECD = sum(ECD .* w) / sum(w);

% manual weighted histogram (stable)
vals = zeros(numel(binEdges)-1,1);
for k = 1:numel(vals)
    inBin = (ECD >= binEdges(k)) & (ECD < binEdges(k+1));
    vals(k) = sum(w(inBin));
end
if sum(vals)>0, vals = vals/sum(vals); end
writematrix(vals(:), docFile, 'Sheet', sheetGS, 'Range', [colLet num2str(2*nBins + 5)]);

%% 10) Summary row
avgEQUIVDia = mean(ECD,'omitnan');
stdEQUIVDia = std(ECD,'omitnan');
avgAR       = mean(AR,'omitnan');
stdAR       = std(AR,'omitnan');

EBSDdata_smDat = [ ...
    avg_maxDimX, std_maxDimX, ...
    avg_maxDimY, std_maxDimY, ...
    avgEQUIVDia, stdEQUIVDia, ...
    avgAR,       stdAR, ...
    meanAreaWeightedECD];

writecell(GSheader, docFile, 'Range', 'L1:T1');
writematrix(EBSDdata_smDat, docFile, 'Range', sprintf('L%d:T%d', n+1, n+1));

fprintf('[%s] DONE. valid grains=%d\n', char(datname), numel(ECD));

end
