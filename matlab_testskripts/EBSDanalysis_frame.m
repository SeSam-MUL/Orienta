%% setup mtex, clear varables and set mtexPath & directions

clear
close all
home

%mtex path - CHANGE & add path for further scripts

mtexpath="C:\Ori_Data\mtex-6.1.0";
addpath(mtexpath);
addpath('C:\Ori_Data\Matlab Skripte');
startup_mtex

% Establish plotting convention - Aztec

setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','intoPlane');

CS=crystalSymmetry('m-3m',[4.05 4.05 4.05], 'mineral', 'Aluminium');
h = Miller({1,0,0},{1,1,0},{1,1,1}, CS);
%%  Input for data analysis -> variables for all analyzed datasets

% where are the files

foldername=['C:\CDL_DePIct-Al\_Wissenschaft\Al Batteries\Al-Mn Al-Mg\h5oina' ...
    '\'];
cd(foldername);

    formatstring='.h5oina'
%
liste = dir([foldername '*' formatstring]); %Laden des gesamten ordners (nur cpr dateien)

files = {liste.name};


%%
% promt= 'Enter the Grain boundary threshold: ';
% gbThreshold = (input(promt))*degree;
% promt= 'Enter the min. number of pixels for 1 grain: ';
% num_pixel = (i2nput(promt));
% promt= 'Enter the min. intercept length in microns: ';
% minLIL=input(promt);
%
% % only with texture analysis
% promt= 'Enter the tolerance angle for texture analysis in degree: ';
% TXTtolAngle=input(promt)*degree;
% promt= 'Apply centerSpecimen for ODF? 1 yes / 0 no';
% autoc=input(promt);
%
% % only with deformation analysis
% promt= 'Enter the tensile Axis as vector (Miller: [x y z]): ';
% loadAxis=input(promt);
% promt= 'Apply inverse rotation for SchmidFactor plots? 1 yes / 0 no';
% SFinv=input(promt);
%
% rot=zeros(length(files), 3);
% disp 'Check/adjust the current matlab folder and the required sample rotation!'
%
% % if some are not rectified
%
% promt= 'apply image rectification? (1=yes; 0=no) ';
% rec = input(promt);
rec=0;

gbThreshold = 4*degree;
num_pixel = 3;
minLIL=1.5;

% only with texture analysis
TXTtolAngle=10*degree;
promt='Should Particles be removed? 1=y; 0=n    '
particleRem=input(promt);

% only with deformation analysis
loadAxis=yvector;
SFinv=1;

promt= 'Apply geometric rotation? 1 yes / 0 no';
geomRot=input(promt);


rot=zeros(length(files), 3);
disp 'Check/adjust the current matlab folder and the required sample rotation!'

% if some are not rectified
% rec = 0;
tc=0;
autoc=1;

%%  predefinition #1

datname=extractBefore(liste(1).name,formatstring);


%%  Define Rotations
n=1;
% for n=1%:length(files)    %


    datname=extractBefore(liste(n).name,formatstring)
    ebsd1=EBSD.load([foldername datname formatstring], 'convertEuler2SpatialReferenceFrame');
    %
    %     % if some are not tilt corrected)
    %     plot(ebsd1);
    %     promt= 'Re-apply tilt correction? (1=yes; 0=no) ';
    %     tc(n) = input(promt);

    tc(n)=0;

    if geomRot==1
        plot(ebsd1);
    end

    % alignment - Check:
    % ori=discreteSample(ebsd1('Aluminium').orientations, 500);
    % figure();
    % plotPDF(ori, h, 'points', 500 , 'MarkerSize', 1.3,  'xAxisdirection', 'south','zAxisdirection', 'IntoPlane', 'grid', 'on')
    %     plotPDF(ebsd1('Aluminium').orientations, h, 'points', 500, 'MarkerSize', 1.3, 'xAxisdirection', 'north','zAxisdirection', 'outOfPlane', 'grid', 'on')
    % drawnow
ori=ebsd1('A').orientations;
    promt='Enter z-Axis rotation in Degree: ';
    rot1=(input(promt));
    r=rotation.byAxisAngle(zvector, rot1*degree);
    ori_R=r*ori;

    % plotPDF(ori_R, h, 'points', 500 , 'MarkerSize', 1.3,  'xAxisdirection', 'south','zAxisdirection', 'IntoPlane', 'grid', 'on')
    %     hold on;  plotPDF(ori_R, h, 'points', 500, 'MarkerSize', 1.3, 'xAxisdirection', 'north', 'add2all');
    % drawnow
    promt='Enter x-Axis rotation in Degree: ';
    rot2=(input(promt));
    r=rotation.byAxisAngle(xvector, rot2*degree);
    ori_R=r*ori_R;
    %
    % plotPDF(ori_R, h, 'points', 1000 , 'MarkerSize', 1.3,  'xAxisdirection', 'south','zAxisdirection', 'IntoPlane', 'grid', 'on')
    % %     hold on;  plotPDF(ori_R, h, 'points', 500, 'MarkerSize', 1.3, 'xAxisdirection', 'north', 'add2all');
    % drawnow
    % promt='Enter z2-Axis rotation in Degree: ';
    % rot2=(input(promt));
    % r=rotation.byAxisAngle(zvector, rot2*degree);
    % ori_R=r*ori_R;
    %
    % plotPDF(ori_R, h, 'points', 1000 , 'MarkerSize', 1.3,  'xAxisdirection', 'south','zAxisdirection', 'IntoPlane', 'grid', 'on')
    % %     hold on;  plotPDF(ori_R, h, 'points', 500, 'MarkerSize', 1.3, 'xAxisdirection', 'north', 'add2all');
    % drawnow
    % promt='Enter x2-Axis rotation in Degree: ';
    % rot2=(input(promt));
    % r=rotation.byAxisAngle(xvector, rot2*degree);
    % ori_R=r*ori_R;

    % plotPDF(ori_R, h, 'points', 500 , 'MarkerSize', 1.3,  'xAxisdirection', 'south','zAxisdirection', 'IntoPlane', 'grid', 'on')
    %     hold on;  plotPDF(ori_R, h, 'points', 500, 'MarkerSize', 1.3, 'xAxisdirection', 'north', 'add2all');
    % drawnow
    promt='Enter y-Axis rotation in Degree: ';
    rot3=(input(promt));
    r=rotation.byAxisAngle(yvector, rot3*degree);
    ori_R=r*ori_R;
    % plotPDF(ori_R, h, 'points', 2000 , 'MarkerSize', 1.3,  'xAxisdirection', 'south','zAxisdirection', 'IntoPlane', 'grid', 'on')
    % drawnow

    %     rot(numrot, 1:3) = [rot1, rot2, rot3];
    rot(n, 1:3) = [rot1, rot2, rot3];

    

    if geomRot==1
        promt='Enter geometric Rotation in Degree: ';
        geomRotAngle(n)=(input(promt));
    else
        geomRotAngle=zeros(1,length(files));
    end
    close all;

    %  plot(ebsd1, ebsd1.bc);
    % colormap gray
    % e=selectPoint(ebsd1);
    % ex(n)=e.x;
    % ey(n)=e.y;
    clearvars ebsd1 promt datname rot1 rot2 rot3 ori_R
    %     numrot=numrot+1;

% end



% nur wenn alle gleich:

for n=2:length(files)
    rot(n, 1:3)=rot(1, 1:3);
end
%% run the data analysis on all datasets in the folder

cd(foldername);


for n=10 % length(files)
    n
   
    home
    datname=extractBefore(liste(n).name,formatstring)
    filename = ['doc_' datname '.xlsx'];

    mkdir(datname);
    if particleRem==1

        CS = { 'notIndexed',...
            crystalSymmetry('m-3m', [4 4 4], 'mineral', 'Aluminium', 'color', [0.53 0.81 0.98]),...
            crystalSymmetry('m-3', [14 14 14], 'mineral', 'Alpha', 'color', [1 0.8 0]),...
            };
        CSA = CS{1,2};
        CST = CS{1,3};
        h = Miller({1,0,0},{1,1,0},{1,1,1}, CSA);
        ph='aluminium';
    else
        ph='indexed';
        CS=crystalSymmetry('m-3m',[4.05 4.05 4.05], 'mineral', 'Aluminium');
    end

    [ebsd_smoothedFill, grains, oM1, r, step, ebsd1, maxX, maxY, ebsd1plot] = ...
    ImportAndModify_MTEX6(formatstring, geomRot, geomRotAngle, rec, tc, ...
                    particleRem, datname, foldername,  ...
                    gbThreshold, num_pixel, rot(n, 1), rot(n, 2), rot(n, 3), n)

   %% BC histogram

% ---------- Settings ----------
docXls   = fullfile(foldername, "Documentation.xlsx");
sheetHist = "BC hist";
sheetFit  = "BC fit (3G)";

binWidth  = 10;
binEdges  = 0:binWidth:230;                     % 0..230
binCenters = binEdges(1:end-1) + binWidth/2;    % 5..225
nBins = numel(binCenters);

% Pre-write bin centers + header label (safe to do every run)
writecell({"BinCenters"}, docXls, "Sheet", sheetHist, "Range", "A1");
writematrix(binCenters(:), docXls, "Sheet", sheetHist, "Range", sprintf("A2:A%d", nBins+1));

% Header for fit-sheet (write once)
writecell({"datname","Nbc", ...
           "BC low w1","mu1","sigma1", ...
           "BC med w2","mu2","sigma2", ...
           "BC high w3","mu3","sigma3"}, ...
          docXls, "Sheet", sheetFit, "Range", "A1");

% ---------- Sheet name ----------
sheetGroup = "BandContrast Group";

% ---------- Write row labels once ----------
rowLabels = { ...
    'BC_high_center', 'BC_high_areaFrac', ...
    'BC_mid_center',  'BC_mid_areaFrac',  ...
    'BC_low_center',  'BC_low_areaFrac'};

writecell([{"Metric"}; rowLabels(:)], docXls, ...
          "Sheet", sheetGroup, "Range", "A1:A7");
% ---------- Loop over datasets ----------


    % ===== Get BC vector from MTEX (make sure it is a vector) =====
    bc = ebsd1.bc(:);                                  % ensure column vector
    bc = bc(isfinite(bc));                       % remove NaN/Inf

    % ===== Histogram counts =====
    counts = histcounts(bc, binEdges, "Normalization","probability").';  % column (nBins x 1)

    % ===== Write to Documentation.xlsx / "BC hist" =====
    colIdx = 2 + (n-1);                          % B=2 for first dataset
    colLet = excelColumn(int32(colIdx));                % e.g., 2->B, 3->C, ...

    % Write header (row 1) and counts (rows 2..nBins+1)
    writecell({char(datname)}, docXls, "Sheet", sheetHist, "Range", sprintf("%s1", colLet));
    writematrix(counts, docXls, "Sheet", sheetHist, "Range", sprintf("%s2:%s%d", colLet, colLet, nBins+1));

    %% ===== Fit 3 Gaussians (GMM) on raw BC =====
    % This is statistically cleaner than fitting Gaussians to histogram bars.
    % Use multiple replicates for stability.
    try
        opts = statset("MaxIter", 2000);
        gm = fitgmdist(bcClamped, 3, ...
            "RegularizationValue", 1e-6, ...
            "Replicates", 10, ...
            "Options", opts);

        w  = gm.ComponentProportion(:);
        mu = gm.mu(:);
        s  = sqrt(squeeze(gm.Sigma(:)));

        % Sort components by mean BC (typical assumption):
        % low BC ~ heavily deformed, mid ~ recovered, high ~ RX
        [mu, idx] = sort(mu, "ascend");
        w = w(idx);
        s = s(idx);

        % Write fit results (one row per dataset)
        row = n + 1;  % because row 1 is header
        writecell({char(datname)}, docXls, "Sheet", sheetFit, "Range", sprintf("A%d", row));
        writematrix(numel(bcClamped), docXls, "Sheet", sheetFit, "Range", sprintf("B%d", row));
        writematrix([w(1) mu(1) s(1) w(2) mu(2) s(2) w(3) mu(3) s(3)], ...
            docXls, "Sheet", sheetFit, "Range", sprintf("C%d:K%d", row, row));

        % (Optional) If you want "fractions" for the three physical classes:
        % Deformed = component 1 (lowest mean), Recovered = component 2, RX = component 3
        % You can store w(1), w(2), w(3) directly or convert to percent.
    catch ME
        warning("3-Gaussian fit failed for %s: %s", string(datname), ME.message);
    end

    % Column for this dataset: B for n=1, C for n=2, ...
colLet = excelColumn(n+1);

% Header (dataset name) in row 1
writecell({char(datname)}, docXls, "Sheet", sheetGroup, "Range", sprintf("%s1", colLet));

% Map components:
% mu(1) low (deformed), mu(2) mid (recovered), mu(3) high (RX)
BC_high_center   = mu(3);
BC_high_areaFrac = w(3);

BC_mid_center    = mu(2);
BC_mid_areaFrac  = w(2);

BC_low_center    = mu(1);
BC_low_areaFrac  = w(1);

vals = [BC_high_center; BC_high_areaFrac; ...
        BC_mid_center;  BC_mid_areaFrac;  ...
        BC_low_center;  BC_low_areaFrac];

% Write values down the column (rows 2..7)
writematrix(vals, docXls, "Sheet", sheetGroup, "Range", sprintf("%s2:%s7", colLet, colLet));
%%

    if geomRot==1
        ebsd1plot=rotate(ebsd1, rotation.byAxisAngle(zvector, geomRotAngle(n)*degree));
        ebsd1plot=ebsd1plot.gridify;
        grainsplot=grains;
        ebsd_smoothedFillplot=ebsd_smoothedFill;      %wurde schon in impMod gedreht mit den Koordinaten 
    else
        ebsd1plot=ebsd1;
        ebsd_smoothedFillplot=ebsd_smoothedFill;
        grainsplot=grains;

    end

    %     figtemp=figure;
    %     plot(ebsd1plot, ebsd1plot.bc);
    %     mtexColorMap black2white
    %     setColorRange('tight');
    %     % hold on
    %     % nextAxis
    %
    %     % für Zugproben, RDy, NDx:
    %     oM1.inversePoleFigureDirection=yvector; %IPFx wrt Z
    %     plot(ebsd1plot(ph),oM1.orientation2color(ebsd1plot(ph).orientations), 'FaceAlpha', 0.75);
    % %     mtexTitle('IPF - colored ND; raw data, bad points removed')
    %     saveas(figtemp, ['IPFrawBC_inRD_' datname '.png']);


    newMtexFigure;
    plot(ebsd1plot, ebsd1plot.bc);
    mtexColorMap black2white
    % setColorRange('tight');
    hold on
    oM1.inversePoleFigureDirection=zvector; % bleibt Datenpunktkoordinatensystem
    plot(ebsd_smoothedFillplot('Aluminium'),oM1.orientation2color(ebsd_smoothedFillplot('Aluminium').orientations), 'FaceAlpha', 0.75);
    % alpha(0.75)
    %     plot(grainsplot.boundary, 'lineColor', 'k', 'linewidth', 0.8);
    %     mtexTitle('IPF - colored ND; smoothed data')
    hold off
    saveFigure(['SmoothedDataBC_inZ_' datname '.png']);
    %         _______________________________________________________________________________________________
    % %             Plot raw vs. smoothed data
    %             filename = ['doc_' datname '.xlsx'];
    %             TxAdditionalInfo={'Number smoothed Grains > numPix Pixel'};
    %             AdditionalData=[length(grains)];
    %
    %             xlswrite(filename,TxAdditionalInfo, datname,'A11:A11');
    %             xlswrite(filename,AdditionalData, datname,'B11:B11');
    %
    %     % __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    %


    %% disp('Grain Size Analysis');

    [EBSDdata_smDat, GSheader] =  GrainSizeAnalysis_mtex6(grains, datname, filename, particleRem, minLIL, step, foldername,n, ebsd1plot, CS);
    close all;
 
    disp('Texture Analysis: works well with >600 "real" grains');
[odfCombined ,VolTexture_odfCombined,VolTextureEBSDsm_fill, drehung] = TextureAnalysis_mtex6(particleRem, TXTtolAngle,ebsd1plot,h,datname, ebsd_smoothedFill,autoc, grains, foldername, n, rot, ph)


    %     disp('Def. Structure');
    %    [kam_smoothed] = DeformationStructure(datname,ebsd1plot,ebsd_smoothedFill, grains);

    % disp('Deformation Analysis');
 [kam_smoothed] = DeformationAnalysis(mtex_path, datname,ebsd_smoothedFill, grains,loadAxis,ebsd1,  SFinv, rot, n , filename, geomRot, ebsd1plot, ph, foldername);    % [kam_smoothed] = DeformationAnalysis(mtex_path, datname,ebsd_smoothedFill, grains,loadAxis,ebsd1,  SFinv, rot, n , filename, geomRot, ebsd1plot, ph, foldername)
    close all;
   
    % disp('RXX Analysis');
    
           [RXedGrainFraction, RXedGrains] = RxxAnalysis(n,foldername, datname,ebsd_smoothedFill, filename, ebsd1)
    
           % %  disp('PSN grains');
% % [grains_PSN,grains_NoPSN, particles, odfPSN, odfNoPSN] =  PSNneighbors(ebsd_goodR, datname, filename,particleRem, ph)

%     SS=specimenSymmetry('mmm')
%     BetaFibrePlot(CS, SS, datname, h, odfCombined);
%
% [CSLlength]=CSLanalysis(ebsd1plot, CS, datname,ph);
%%
% 
% close all;
% figtemp=figure;
% plot(ebsd1plot, ebsd1plot.bc, 'figSize','huge')
% % plot(ebsd1plot, ebsd1plot.bc)
% 
% mtexColorMap black2white
% % setColorRange([0 220])
% saveas(figtemp, ['BC' datname '.png']);
% hold on
% try
%     plot(ebsd1plot(ebsd1plot.prop.Si_Ka1>(0.35*max(max(ebsd1plot.prop.Si_Ka1)))), 'FaceColor', 'DodgerBlue', 'FaceAlpha', 0.65, 'DisplayName', 'Si')
% end 
% try
%     plot(ebsd1plot(ebsd1plot.prop.Mn_Ka1>(0.35*max(max(ebsd1plot.prop.Mn_Ka1)))), 'FaceColor', 'FireBrick', 'FaceAlpha', 0.65, 'DisplayName', 'Mn')
% end
% try
%     plot(ebsd1plot(ebsd1plot.prop.Fe_Ka1>(0.22*max(max(ebsd1plot.prop.Fe_Ka1)))), 'FaceColor', 'Red', 'FaceAlpha', 0.65, 'DisplayName', 'Fe')
% end
% try
%     plot(ebsd1plot(ebsd1plot.prop.Cu_La1_2>(0.30*max(max(ebsd1plot.prop.Cu_La1_2)))), 'FaceColor', 'DarkViolet', 'FaceAlpha', 0.55, 'DisplayName', 'Cu')
% end
% try
%     plot(ebsd1plot(ebsd1plot.prop.Zn_La1_2>(0.35*max(max(ebsd1plot.prop.Zn_La1_2)))), 'FaceColor', 'Gold', 'FaceAlpha', 0.55, 'DisplayName', 'Zn')
% end
% try
%         plot(ebsd1plot(ebsd1plot.prop.Cu_Ka1>(0.30*max(max(ebsd1plot.prop.Cu_Ka1)))), 'FaceColor', 'Fuchsia', 'FaceAlpha', 0.55, 'DisplayName', 'Cu')
% end
% try
%     plot(ebsd1plot(ebsd1plot.prop.Zn_Ka1>(0.35*max(max(ebsd1plot.prop.Zn_Ka1)))), 'FaceColor', 'Gold', 'FaceAlpha', 0.55, 'DisplayName', 'Zn')
% end
% try
%     plot(ebsd1plot(ebsd1plot.prop.Sn_La1>(0.35*max(max(ebsd1plot.prop.Sn_La1)))), 'FaceColor', 'MidnightBlue', 'FaceAlpha', 0.65, 'DisplayName', 'Sn')
% end
% try
%     plot(ebsd1plot(ebsd1plot.prop.Mg_Ka1_2>(0.270*max(max(ebsd1plot.prop.Mg_Ka1_2)))), 'FaceColor', 'SteelBlue', 'FaceAlpha', 0.55, 'DisplayName', 'Mg')
% end
% 
% % plot(grains('Alpha'), 'FaceColor', 'Tomato', 'FaceAlpha', 0.55)
% % legend('off')
% saveas(figtemp, ['BC_particles_Elements' datname '.png']);

%%

% wenn nur TXT Nachbarn angesehen werden:
%             save(['variables' datname '.mat'],'grains','CS', 'h', 'gbThreshold','num_pixel','datname','ebsd1', 'filename', 'ebsd_smoothedFill','grains', '-v7.3')%)%,'drehung', 'odfCombined','particles', ,'grains_PSN','grains_NoPSN' 'kam_smoothed','GSheader','psiMod','RXedGrainFraction', 'gBC_smoothed', 'gid_smoothed','gkam', 'gid')% ,'cells', 'loadAxis',)%,
rotInitial=rot(n,:);

save(['variables' datname '.mat'], 'ebsd_smoothedFill','grainsplot','step','ebsd1plot','particleRem','rotInitial', 'grains','CS', 'h', 'gbThreshold','num_pixel','datname','ebsd1', 'filename')
try
    save(['variables' datname '.mat'], 'kam_smoothed','SAGBlength','HAGBlength', '-append')
end
try
    save(['variables' datname '.mat'],'drehung', 'odfCombined', '-append')
end
try
    save(['variables' datname '.mat'], 'grains_PSN','grains_NoPSN', 'particles', 'odfPSN', 'odfNoPSN', '-append')
end
try
    save(['variables' datname '.mat'],'RXedGrainFraction', '-append')
end
try
    save(['variables' datname '.mat'], 'kam_smoothed', '-append')
end
clearvars a drehung b rotinitial odf_sy odfCombined odfNoPSN odfPSN ori oriNoPSN oriPSN p_ids idsPSN step particles grains_PSN grains_NoPSN datname cells ebsd1 ebsd_good ebsd_goodR ebsd_goodR1 ebsd_smoothedFill odfR gBC_smoothed gid gid_smoothed gkam grains grains_smoothed kam_goodR kam_smoothed namesVolTextureODFR RXedGrainFraction SF VolTextureEBSDsm_fill VolTextureODFR rot1 rot2 rot3  EBSDdata_smDat GSheader odfMagicRule psiMagicRule psiMod psiGrains odf_grainKernel odf_kernelMod oM1 filename condition numparts VolTexture_odfCombined
close all

cd ..
    % end
end

disp 'finished'


%% extract dependencies
% Define the function folder and name
functionFolder = ['C:\Ori_Data\Matlab Skripte'];
% functionName = 'EBSDanalysis_frame_2Phase.m';
functionName = 'EBSDanalysis_frame.m';


% Change directory to the folder containing the function
cd(functionFolder)

% Get the list of dependencies
fList = matlab.codetools.requiredFilesAndProducts(fullfile(functionFolder, functionName));

% Save the list of dependencies to a text file
fileID = fopen('dependencies.txt', 'w');
fprintf(fileID, '%s\n', fList{:});
fclose(fileID);

% Create a directory for saving the function and its dependencies
outputDir = [foldername 'MyFunctionWithDependencies'];
mkdir(outputDir);

% Copy the function file itself
copyfile(fullfile(functionFolder, functionName), outputDir);

% Define the path to exclude
excludePath = mtex_path;

% Copy each dependent file to the new directory, excluding the specified path
for i = 1:length(fList)
    if ~startsWith(fList{i}, excludePath)
        copyfile(fList{i}, outputDir);
    end
end

disp('Extracted dependencies');
