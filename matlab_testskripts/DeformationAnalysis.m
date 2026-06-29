function [kam_smoothed] = DeformationAnalysis(mtex_path, datname,ebsd_smoothedFill, grains,loadAxis,ebsd1,  SFinv, rot, n , filename, geomRot, ebsd1plot, ph, foldername)

%% GB-misorientation map
close all;
% [grains_sm,ebsd_smoothedFill.grainId,ebsd_smoothedFill.mis2mean] = calcGrains(ebsd_smoothedFill(ph),'alpha', 1.5, 'angle', 2*degree);
[grains_sm,ebsd_smoothedFill.grainId,ebsd_smoothedFill.mis2mean] = calcGrains(ebsd_smoothedFill,'alpha', 2.5, 'angle', 2*degree);
grains_sm=smooth(grains_sm,3);
GBMiso=grains_sm.boundary(ph,ph);

% GBInnerMiso=grains_sm.innerBoundary;
SAGB=GBMiso(GBMiso.misorientation.angle<15*degree,GBMiso.misorientation.angle>3*degree);
SAGBlength=sum(SAGB.segLength);
HAGB=GBMiso(GBMiso.misorientation.angle>15*degree);
HAGBlength=sum(HAGB.segLength);
GBlengthtot=sum(GBMiso.segLength);

if strcmp(mtex_path, 'C:\Ori_Data\mtex-6.1.0')==1
    AreaEBSDAl=(2*ebsd_smoothedFill.unitCell.x(1))^2 * length(ebsd_smoothedFill(ph)); %
else
    AreaEBSDAl=(2*ebsd_smoothedFill.unitCell(1))^2 * length(ebsd_smoothedFill(ph));
end

dimX = ebsd_smoothedFill.extent(2)-ebsd_smoothedFill.extent(1);
dimY = ebsd_smoothedFill.extent(4)-ebsd_smoothedFill.extent(3);

GBperEBSD_sf_area=GBlengthtot/AreaEBSDAl
GBperfieldArea=GBlengthtot/dimX/dimY

HAGBperEBSD_sf_area=HAGBlength/AreaEBSDAl
HAGBperfieldArea=HAGBlength/dimX/dimY
LAGBperEBSD_sf_area=SAGBlength/AreaEBSDAl
LAGBperfieldArea=SAGBlength/dimX/dimY


    docFile = fullfile(foldername, 'Documentation.xlsx');
    sheetname= "Grain Boundaries";
     colLet = excelColumn(n+2);
     
     writecell({char(datname)}, docFile, "Sheet", sheetname, "Range",sprintf("%s1", colLet));
    writecell([{'GBperEBSD Al_area'; 'GBperfieldArea'; 'HAGBperEBSD Al_area'; ...
    'HAGBperfieldArea'; 'LAGBperEBSD Al_area'; 'LAGBperfieldArea'}], docFile, "Sheet",sheetname, "Range", "A3:A9");
   

    data = [GBperEBSD_sf_area; GBperfieldArea; HAGBperEBSD_sf_area; ...
    HAGBperfieldArea; LAGBperEBSD_sf_area; LAGBperfieldArea];
writecell(num2cell(data),docFile, 'Sheet',sheetname, 'Range', sprintf("%s3", colLet));

Sphericity = grains.area ./ grains.perimeter('withInclusion') ./ grains.equivalentRadius;
writecell({'sphericity'}, docFile, 'Sheet', sheetname, 'Range', 'A10');
writecell(num2cell(Sphericity), docFile, 'Sheet', sheetname, ...
    'Range',  sprintf("%s10", colLet));
try
    save(['variables' datname '.mat'], 'GBperEBSD_sf_area','GBperfieldArea','HAGBperEBSD_sf_area','HAGBperfieldArea','LAGBperEBSD_sf_area','LAGBperfieldArea', '-append')

end

%% 40°<111> Grain boundaries between smoothed grains drawn in red
%  rotGB = rotation.byAxisAngle(vector3d(1,1,1),40*degree); % Definition of GB
% % rotGB=unique(symmetrise(rotGB, CS))
%  gbA=grains.boundary(ph,ph);     % consider only boundary segments between indexed grains
%  ind = angle(GBMiso.misorientation,rotGB)<5*degree;     % max. angular deviation 5°
% newMtexFigure;
% plot(ebsd1plot, ebsd1plot.bc)
% mtexColorMap black2white
% hold on
% plot(grains.boundary, 'lineColor','DarkBlue');% hold on
% %  plot(GBMiso);
%  hold on
%
% plot(GBMiso(ind), 'lineWidth', 1.2, 'lineColor','Tomato');
% % mtexTitle('Mobile boundaries');
%  annotation('textbox', 'String', {'\color[rgb]{0.6,0.6,0.6} all boundaries', ...
%      '\color[rgb]{0,0,0} Al-Al boundaries', '\color[rgb]{1,0,0} 40° <111> boundaries'},...
%      'BackgroundColor',[1,1,1], 'Position',[0.7, 0.05, 0.1, 0.1], 'FitBoxToText','on');
%  hold off
%  saveFigure(['40degGrainboundaries_' datname '.png']); %saveas eingefügt
%
%  %             filename = ['documentation_' datname '.xlsx'];
%  fRot40GB=length(GBMiso(ind))/length(GBMiso)*100;
%  xlswrite(filename,{'Fraction 40 ° boundaries in %'},datname,'A12:A12');
%  xlswrite(filename,fRot40GB,datname,'B12:B12');

%%
GBmapMiso=figure;
plot(ebsd1plot, ebsd1plot.bc)
colormap gray
freezeColors;
hold on
plot(GBMiso,GBMiso.misorientation.angle./degree, 'linewidth',2);
set(gca,'CLim',[1 63])  %,'ColorScale','log')
colormap parula
freezeColors;
hold on
plot(grains_sm.innerBoundary, 'LineColor', 'DarkGray','linewidth',0.8)
hold on
plot(grains_sm.boundary(ph,'n'), 'LineColor', 'k','linewidth',1)
% plot(grains_sm('n'), 'FaceColor', 'Black', 'FaceAlpha', 0.8)
mtexTitle('Grain boundary misorientation');
mtexColorbar('title','Misorientation angle');
saveas(GBmapMiso, ['GBmisorientation_' datname '.png']);
close all

% seg length GB miso
seglengths=zeros(32,1);
j=1;
for i=3:2:63
    segs = GBMiso((i) < GBMiso.misorientation.angle/degree, GBMiso.misorientation.angle/degree < (i+2));
    seglengths(j)=sum(segs.segLength);
    j=j+1;
end
headers = { 'sum GB seg.lengths'};
writecell(headers, docFile, 'Sheet', sheetname, 'Range', 'A11:A41');
writecell(num2cell((3:2:63).'), docFile, 'Sheet', sheetname, 'Range', 'B11:B41');
writecell(num2cell(seglengths), docFile, 'Sheet', sheetname, 'Range', sprintf("%s11", colLet));


% %standard mtex GB miso
% figtemp=figure;
% GBhist=histogram([(angle(GBMiso.misorientation)./degree)],'BinWidth', 2)%, 'Normalization', 'probability');
% mtexTitle('GB misorientation')
% saveas(figtemp, ['GBmisorientationHistogram_' datname '.png']);
% % xlswrite(filename,[{'GB misorientation Angle'},{'norm. fraction'}],datname,'K28:L28')
% xlswrite(filename,[{'GB misorientation Angle'},{'abs counts'}],datname,'K28:L28')
% xlswrite(filename,(GBhist.BinEdges+1)',datname,'K29:K61')
% xlswrite(filename,GBhist.Values',datname,'L29:L61')
% close all
%
% figtemp=figure;
% histogram(GBMiso.misorientation.angle./degree, 'BinWidth', 1,'BinLimits',[0,64]);
% xlabel('GB misorientation in deg');
% saveas(figtemp, ['GBmisorientationHist_' datname '.png']);

% subGB=GBMiso.misorientation.angle/degree;
% subGB(subGB<0.8)=[];
% subGB(subGB>15)=[];
% mean(subGB)
% std(subGB)
% GBmiso=GBMiso.misorientation.angle./degree;
%% KAM
close all
order=2;
Threshold=7;
kam_smoothed = KAM(ebsd_smoothedFill,'threshold',Threshold*degree, 'order', order)./degree;
[KAMhist_smoothed_Counts, KAMhist_smoothed_Edges]=histcounts(kam_smoothed, 'BinWidth', 0.25,'normalization', 'probability');

docFile = fullfile(foldername, 'Documentation.xlsx');
sheetKAMHist = "KAM hist";
writecell([{"Order"}; {"Threshold"}], docFile, "Sheet", sheetKAMHist, "Range", "A1:A2");
writematrix([order;Threshold], docFile, 'Sheet', sheetKAMHist, 'Range', "B1:B2");

% Column for this dataset: B for n=1, C for n=2, ...
colLet = excelColumn(n+1);
% Header (dataset name) in row 1
writecell({char(datname)}, docFile, "Sheet", sheetKAMHist, "Range", sprintf("%s3", colLet));
writecell({"BinCenters"}, docFile, "Sheet", sheetKAMHist, "Range", "A4");
writecell(num2cell(KAMhist_smoothed_Edges(:)), docFile, 'Sheet', sheetKAMHist, 'Range', 'A5');
writematrix([KAMhist_smoothed_Counts(:)], docFile, 'Sheet', sheetKAMHist, 'Range', sprintf("%s5", colLet));


% für Belinda
%             kam_smoothed=kam_smoothed(~isnan(kam_smoothed));
%
%             range0_05=kam_smoothed(kam_smoothed<0.5*degree);
%             range05_1=kam_smoothed(kam_smoothed>=0.5*degree & kam_smoothed<1  *degree);
%             range1_15=kam_smoothed(kam_smoothed>=1  *degree & kam_smoothed<1.5*degree);
%             range15_2=kam_smoothed(kam_smoothed>=1.5*degree & kam_smoothed<2  *degree);
%             range2_25=kam_smoothed(kam_smoothed>=2  *degree & kam_smoothed<=2.5 *degree);
%             range25_3=kam_smoothed(kam_smoothed>=2.5  *degree & kam_smoothed<=3 *degree);
%
%             length(kam_smoothed);
%             absVar=[length(range0_05) length(range05_1) length(range1_15) length(range15_2) length(range2_25) length(range25_3)]
%             relVar=absVar./length(kam_smoothed)
%% KAM maps

% KAMfig=figure;
% plot(ebsd1plot, ebsd1plot.bc, 'figSize','huge')
% colormap(b2wcontrast)
% setColorRange([30 180])
% freezeColors
% hold on
% out=ebsd1plot(ebsd1plot.prop.Al_Ka1< 14000)
% plot(out, out.bc)
% setColorRange([60 200])
% freezeColors
% hold on
% plot(ebsd_smoothedFill(ph), kam_smoothed./ degree, 'FaceAlpha', 'flat', 'FaceVertexAlphaData', 0.85);
% hold on
% colormap(KAM_WhiteBlueHeatColorMap)
% setColorRange([0 3])
% hold on
% mtexColorbar;
% 
% plot(grains(grains.numPixel>10).boundary('A', 'A'), 'lineColor', 'DarkSlateGray', 'lineWidth', 1);
% saveas(KAMfig, ['KAM_wbh_' datname '.png']);
% 
% %
% % hold on
% % try
% % plot(ebsd1plot(ebsd1plot.prop.meanFeMn>20000), 'FaceColor', 'LightBlue', 'FaceAlpha', 0.65)
% % legend('off')
% % end
% % try
% % plot(ebsd1plot(ebsd1plot.prop.Zn_La1_2>20000), 'FaceColor', 'LightBlue', 'FaceAlpha', 0.65)
% % legend('off')
% % end
% % saveas(KAMfig, ['KAM_lajolla' datname '.png']);
% 
% % colormap(batlowWS);
% % saveas(KAMfig, ['KAM_batlowWS' datname '.png']);
% % colormap(viridis);
% % hold on
% %                                         % plot(grains('notIndexed'), 'FaceColor','k', 'FaceAlpha', 0.3)
% % hold on
% %             try
% %             plot(grains('Alpha'), 'FaceColor', 'Tomato', 'FaceAlpha', 0.65)
% %             end
% % hold on
% % plot(grains.boundary('Aluminium'), 'lineColor','black','lineWidth',0.75)
% % plot(grains.boundary('indexed', 'indexed'), 'lineColor','DarkGray','lineWidth',2)
% % try
% % hold on
% % plot(grains('T'), 'FaceColor', 'Moccasin', 'FaceAlpha', 0.45, 'lineColor','none')
% hold on
% 
% try
%     plot(ebsd1plot(ebsd1plot.prop.Si_Ka1>(0.25*max(max(ebsd1plot.prop.Si_Ka1)))), 'FaceColor', 'Gold', 'FaceAlpha', 0.65, 'DisplayName', 'Si')
% end
% % try
% %     plot(ebsd1plot(ebsd1plot.prop.Mn_Ka1>(0.35*max(max(ebsd1plot.prop.Mn_Ka1)))), 'FaceColor', 'FireBrick', 'FaceAlpha', 0.65, 'DisplayName', 'Mn')
% % end
% try
%     plot(ebsd1plot(ebsd1plot.prop.Fe_Ka1>(0.25*max(max(ebsd1plot.prop.Fe_Ka1)))), 'FaceColor', 'Tomato', 'FaceAlpha', 0.65, 'DisplayName', 'Fe')
% end
% try
%     plot(ebsd1plot(ebsd1plot.prop.Cu_La1_2>(0.35*max(max(ebsd1plot.prop.Cu_La1_2)))), 'FaceColor', 'Wheat', 'FaceAlpha', 0.65, 'DisplayName', 'Cu')
% end
% % try
% %     plot(ebsd1plot(ebsd1plot.prop.Zn_La1_2>(0.35*max(max(ebsd1plot.prop.Zn_La1_2)))), 'FaceColor', 'DarkOrange', 'FaceAlpha', 0.65, 'DisplayName', 'Zn')
% % end
% % try
% %         plot(ebsd1plot(ebsd1plot.prop.Cu_Ka1>(0.35*max(max(ebsd1plot.prop.Cu_Ka1)))), 'FaceColor', 'Wheat', 'FaceAlpha', 0.65, 'DisplayName', 'Cu')
% % end
% % try
% %     plot(ebsd1plot(ebsd1plot.prop.Zn_Ka1>(0.35*max(max(ebsd1plot.prop.Zn_Ka1)))), 'FaceColor', 'Chocolate', 'FaceAlpha', 0.65, 'DisplayName', 'Zn')
% % end
% % try
% %     plot(ebsd1plot(ebsd1plot.prop.Sn_La1>(0.35*max(max(ebsd1plot.prop.Sn_La1)))), 'FaceColor', 'LightCyan', 'FaceAlpha', 0.65, 'DisplayName', 'Sn')
% % end
% try
%     plot(ebsd1plot(ebsd1plot.prop.Mg_Ka1_2>(0.40*max(max(ebsd1plot.prop.Mg_Ka1_2)))), 'FaceColor', 'DeepPink', 'FaceAlpha', 0.65, 'DisplayName', 'Mg')
% end
% legend ('Location', 'eastoutside')
% 
% saveas(KAMfig, ['KAM_wbh_EDS_' datname '.png']);
% close all

KAMfig=figure;
plot(ebsd1plot, ebsd1plot.bc, 'figSize','huge')%, 'micronbar', 'off')
% plot(ebsd1plot, ebsd1plot.prop.quality, 'figSize','huge')
colormap(b2wcontrast)
setColorRange([20 200])
freezeColors
hold on
try
out=ebsd1plot(ebsd1plot.prop.Al_Ka1< 14000)
plot(out, out.bc)
setColorRange([60 200])
freezeColors
hold on
end
plot(ebsd_smoothedFill, kam_smoothed, 'FaceAlpha', 0.8);
hold on
setColorRange([0 2])
mtexColorbar;
colormap(viridis)
hold on
plot(grains_sm(grains_sm.numPixel>10).boundary(ph,ph), 'lineColor', 'Black', 'lineWidth', 0.8, 'Alpha', 0.8);
saveas(KAMfig, ['KAM_viridisSmooth_' datname '.png']);
% try
% hold on
% plot(ebsd1plot(ebsd1plot.prop.Al_Ka1<(0.2*max(max(ebsd1plot.prop.Al_Ka1))), ebsd1plot.bc<45), 'FaceColor', 'Black', 'FaceAlpha', 0.7, 'micronbar', 'off')
% end
% %         figtemp = figure; plot(ebsd1plot, ebsd1plot.bc);
% %         polyoutside = selectPolygon
% %         edsData = ebsd1plot(~inpolygon(ebsd1plot,polyoutside))
% % 
% % disp('Please close the image to proceed.');
% % uiwait(figtemp);
% % edsData=ebsd_smoothedFill;
% try
%     plot(edsData(edsData.prop.Si_Ka1>(0.16*max(max(edsData.prop.Si_Ka1)))), 'FaceColor', 'Gold', 'FaceAlpha', 0.65, 'DisplayName', 'Si')
% end
% 
% try
%     plot(edsData(edsData.prop.Fe_Ka1>(0.16*max(max(edsData.prop.Fe_Ka1)))), 'FaceColor', 'Red', 'FaceAlpha', 0.69, 'DisplayName', 'Fe')
% end
% try
%     % plot(edsData(edsData.prop.Cu_La1_2>(0.35*max(max(edsData.prop.Cu_La1_2)))), 'FaceColor', 'DarkSeaGreen', 'FaceAlpha', 0.65, 'DisplayName', 'Cu')
% end
% try
%     % plot(ebsd1plot(ebsd1plot.prop.Mn_Ka1>(0.30*max(max(ebsd1plot.prop.Mn_Ka1)))), 'FaceColor', 'DarkViolet', 'FaceAlpha', 0.65, 'DisplayName', 'Mn')
% end
% % try
% %     plot(edsData(edsData.prop.Zn_La1_2>(0.35*max(max(edsData.prop.Zn_La1_2)))), 'FaceColor', 'DarkOrange', 'FaceAlpha', 0.65, 'DisplayName', 'Zn')
% % end
% % try
%         % plot(edsData(edsData.prop.Cu_Ka1>(0.35*max(max(edsData.prop.Cu_Ka1)))), 'FaceColor', 'Wheat', 'FaceAlpha', 0.65, 'DisplayName', 'Cu')
% % end
% % try
% %     plot(edsData(edsData.prop.Zn_Ka1>(0.35*max(max(edsData.prop.Zn_Ka1)))), 'FaceColor', 'Chocolate', 'FaceAlpha', 0.65, 'DisplayName', 'Zn')
% % end
% % try
% %     plot(edsData(edsData.prop.Sn_La1>(0.35*max(max(edsData.prop.Sn_La1)))), 'FaceColor', 'LightCyan', 'FaceAlpha', 0.65, 'DisplayName', 'Sn')
% % end
% try
%     plot(edsData(edsData.prop.Mg_Ka1_2>(0.32*max(max(edsData.prop.Mg_Ka1_2)))), 'FaceColor', 'Lime', 'FaceAlpha', 0.65, 'DisplayName', 'Mg')
% end
% % legend ('Location', 'eastoutside')
% legend ('off')
% 
% 
% % legend('off')
% saveas(KAMfig, ['KAM_viridisSmooth_EDS_' datname '.png']);

%%
close all
%%         %
%%          Schmid factor calculation
%
%         %to avoid lare area averaging - create new, small seperated grains
%         %     [cells,ebsd_smoothedFill.grainId, ebsd_smoothedFill.mis2mean]=calcGrains(ebsd_smoothedFill,'angle',3*degree,'boundary' ,'tight');
%
%         % test schmid mit ebsd1
%         cells=calcGrains(ebsd1, 'alpha', 2.5, 'angle', 2.5*degree)
%         % cells=grains_sm;
%         % cells(cells.numPixel<20)=[];
%         cells=cells(cells(ph));
%
%         %% Local SF
%         sS = slipSystem.fcc(ebsd_smoothedFill(ph).CS);
%         sS = sS.symmetrise;%('antipodal');
%         loadAxis=xvector;
%         sigma = stressTensor.uniaxial(loadAxis);
%
%         % rotate the stress tensor into crystal coordinates
%
%         % sigmaLocal = inv(ebsd_smoothedFill(ph).orientations) * sigma;
%         sigmaLocal = inv(ebsd_smoothedFill(ph).orientations) * sigma;
%
%
%         % the resulting matrix is the same as above
%         SF = sS.SchmidFactor(sigmaLocal);
%         [SFMax,active] = max(SF,[],2);
%
%    figtemp=figure;
%         plot(ebsd1plot, ebsd1plot.bc)
%         colormap gray
%         freezeColors
%         hold on
%         plot(ebsd_smoothedFill(ph), SFMax);
% alpha(0.5);
%         hold on
%         plot(cells(ph).boundary, 'lineColor', [0.4,0.4,0.4], 'lineWidth', 0.5);
%         setColorRange('tight');
%         colormap(b2rI)    %modblueyelloworange;
%         mtexTitle(['Local max. Schmid Factor, x=' num2str(loadAxis.x) ' y=' num2str(loadAxis.y)' ' z=' num2str(loadAxis.z)]);
%         mtexColorbar
%         saveas(figtemp, ['LocalmaxSFs_' datname '.png']);
%
%         % second slip system
%         SFmod=SF;
%         for i=1:length(SFMax)
%             SFmod(i,active(i))=NaN;
%         end
%         [SF2nd,active2] = max(SFmod,[],2);
%
%         % third slip system
%         for i=1:length(SFMax)
%             SFmod(i,active2(i))=NaN;
%         end
%         [SF3rd,~] = max(SFmod,[],2);
%
%         sf123=(SFMax.*SF2nd.*SF3rd).^1/3;
%
%
%         figtemp=figure;
%         plot(ebsd1plot, ebsd1plot.bc)
%         colormap gray
%         freezeColors
%         hold on
% %      plot(ebsd_smoothedFill(ph), kam_smoothed./ degree, 'FaceAlpha', 'flat', 'FaceVertexAlphaData', 0.7);
% % colormap(KAM_WhiteBlueHeatColorMap)
% %         setColorRange([0.0,1.5]);
% % freezeColors
%
%
%
% % plot(ebsd1plot(ph), SFMax);
% plot(ebsd_smoothedFill(ph), sf123);
%
% alpha(0.5);
%         hold on
%         plot(cells(ph).boundary, 'lineColor', [0.4,0.4,0.4], 'lineWidth', 0.5);
%         setColorRange('tight');
%         colormap(b2rI)    %modblueyelloworange;
%         mtexTitle(['Local max. Schmid Factor, x=' num2str(loadAxis.x) ' y=' num2str(loadAxis.y)' ' z=' num2str(loadAxis.z)]);
%         mtexColorbar
%         saveas(figtemp, ['Localfirst3SFs_' datname '.png']);
%
%
%
% %          %%  __________________________________
% %
% %         % SFinv=1: % wenn nur mit keepXY gedreht wurde: die Pfeile sollen dann
% %         % richtig ins Bild gedreht werden
% %         % wenn nur teilweise keepXY angewandt wird (=geomRot=1) muss nur der keepXY
% %         % Teil rückgedreht werden
% %
% %         % SFinv=0: % wenn nicht oder ganz ohne keepXY gedreht wurde: Bild
% %         % entspricht Orientieurngen
% %         % loadAxis in Probenkoordinaten
% %
% %         rot1=rotation.byAxisAngle(zvector, rot(n, 1)*degree);
% %         rot2=rotation.byAxisAngle(xvector, rot(n, 2)*degree);
% %         rot3=rotation.byAxisAngle(yvector, rot(n, 3)*degree);
% %
% %         if SFinv==1
% %             if geomRot==0       % wenn gedreht, dann immer mit keepXY
% %
% %         r=(rot3*(rot2*rot1));
% %
% %             cells.meanOrientation=inv(r).*(cells.meanOrientation);
% %             ebsd_smoothedFill(ph).orientations=inv(r).*(ebsd_smoothedFill(ph).orientations);
% %
% %         %     %optional mis2mean plot
% %         %     figtemp=figure;
% %         %     plot(ebsd_smoothedFill, ebsd_smoothedFill.mis2mean.angle./degree);
% %         %     setColorRange([0, 10]);
% %         %     mtexColorbar('Title', 'Misorientation to mean grain orientation [deg]')
% %         %     hold on
% %         %     plot(cells.boundary, 'lineColor', 'k')
% %         %     hold off
% %         %     saveas(figtemp, ['mis2mean_' datname '.png']); %saveas eingefügt
% %
% %         sS = slipSystem.fcc(ebsd_smoothedFill(ph).CS);
% %         sS = sS.symmetrise('antipodal');
% %         sigma = stressTensor.uniaxial(loadAxis);
% %         % rotate the stress tensor into crystal coordinates
% %         sigmaLocal = inv(cells.meanOrientation) * sigma;
% %         SF = sS.SchmidFactor(sigmaLocal);
% %
% %         % take the maxium along the rows
% %         [SFMax,active] = max(SF,[],2);
% %
% %         % second slip system
% %         SFmod=SF;
% %         for i=1:length(SFMax)
% %             SFmod(i,active(i))=NaN;
% %         end
% %         [~,active2] = max(SFmod,[],2);
% %
% %         % third slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active2(i))=NaN;
% %         end
% %         [~,active3] = max(SFmod,[],2);
% %
% %         % 4th slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active3(i))=NaN;
% %         end
% %         [~,active4] = max(SFmod,[],2);
% %
% %         % 5th slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active4(i))=NaN;
% %         end
% %         [~,active5] = max(SFmod,[],2);
% %
% %         % 6th slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active5(i))=NaN;
% %         end
% %         [~,active6] = max(SFmod,[],2);
% %         %__________________________________________________________________________________________________
% %         %% create the Schmid factor plots
% %         % figtemp=figure;
% %         % plot(cells(ph).boundary);
% %         % hold on
% %         % plot(cells(ph),SFMax)
% %         % mtexColorbar
% %         % setColorRange([0.2,0.5]);
% %         % colormap modblueyelloworange;
% %         % mtexTitle('max. Schmid Factor')
% %         % saveas(figtemp, ['SFs_' datname '.png']);
% %         % close all
% %
% %         % take the active slip system and rotate it in specimen coordinates
% %         sSactive = cells(ph).meanOrientation .* sS(active);
% %
% %         figtemp=figure;
% % plot(ebsd1plot, ebsd1plot.bc)
% % colormap gray
% % freezeColors;
% % hold on
% %         plot(ebsd_smoothedFill(ph), kam_smoothed./degree);
% %         hold on
% %         plot(grains.boundary, 'lineColor','k', 'LineWidth', 1.2);
% %         setColorRange([0,1.5])
% % colormap(KAM_WhiteBlueHeatColorMap)
% %         mtexTitle('Slip Systems highest 6 SFs')
% %         hold on
% %
% %
% %
% %             % calc and plot the slip plane of the 6th  max. slip system
% %             sSactive6 = cells(ph).meanOrientation .* sS(active6);
% %             quiver(cells(ph),sSactive6.trace,'color',[0.3 0.3 0.3],'linewidth', 0.5, 'lineStyle', ':');
% %             % and the slip direction of the 6th  max. slip system
% %             %     quiver(cells(ph),sSactive6.b,'color',[0.3 0.3 0.3],'linewidth',  0.5, 'lineStyle', ':');
% %
% %             % calc and plot the slip plane of the 5th  max. slip system
% %             sSactive5 = cells(ph).meanOrientation .* sS(active5);
% %             quiver(cells(ph),sSactive5.trace,'color',[0.3 0.3 0.3],'linewidth',  0.5, 'lineStyle', ':');
% %             % and the slip direction of the 5th  max. slip system
% %             %     quiver(cells(ph),sSactive5.b,'color',[0.3 0.3 0.3],'linewidth',  0.5, 'lineStyle', ':');
% %
% %             % calc and plot the slip plane of the 4th  max. slip system
% %             sSactive4 = cells(ph).meanOrientation .* sS(active4);
% %             quiver(cells(ph),sSactive4.trace,'color',[0.5 0.5 0.3],'linewidth', 1);
% %             % and the slip direction of the 4th  max. slip system
% %             %     quiver(cells(ph),sSactive4.b,'color',[0.5 0.5 0.3],'linewidth', 1);
% %
% %             %the third  max. slip system
% %             sSactive3 = cells(ph).meanOrientation .* sS(active3);
% %             quiver(cells(ph),sSactive3.trace,'color',[0.5 0.5 1],'linewidth', 1);
% %             %     quiver(cells(ph),sSactive3.b,'color',[0.5 0.5 1],'linewidth', 1);
% %
% %             %the second max. slip system
% %             sSactive2 = cells(ph).meanOrientation .* sS(active2);
% %             quiver(cells(ph),sSactive2.trace,'color',[0.3,0.8,0.4],'linewidth', 1);
% %             %     quiver(cells(ph),sSactive2.b,'color',[0.3,0.8,0.4],'linewidth', 1);
% %
% %             % the maximum slip system
% %             quiver(cells(ph),sSactive.trace,'color',[0.9,0.2,0.4]);
% %             %     quiver(cells(ph),sSactive.b,'color',[0.9,0.2,0.4]);
% %
% %             %     legend('6th Highest SF sS', '','5th Highest SF sS', '','4th Highest SF sS', '', '3rd Highest SF sS', '','2nd Highest SF sS', '','Highest SF sS', '')
% %             legend('6th Highest SF sS', '5th Highest SF sS', '4th Highest SF sS',  '3rd Highest SF sS', '2nd Highest SF sS', 'Highest SF sS')
% %
% %             %plot(cells.boundary, 'color', 'k');
% %             hold off
% %             saveas(figtemp, ['6HighestSF-sS_' datname '.png']);
% %             %     close all
% %
% %           % das heißt wenn die Orientierungen aus dem plot-Koordinatensystem rausgedreht wurden
% %             %müssen die schmidfaktorpfeile zurückgedreht werden für den Plot
% %
% %         elseif geomRot==1
% %
% %                 r=rotation.byAxisAngle(yvector, rot(n,3)*degree)*rotation.byAxisAngle(xvector, rot(n,2)*degree)
% %
% %             cells(ph).meanOrientation=inv(r).*(cells(ph).meanOrientation);
% %             ebsd_smoothedFill(ph).orientations=inv(r).*(ebsd_smoothedFill(ph).orientations);
% %
% %         sS = slipSystem.fcc(ebsd_smoothedFill(ph).CS);
% %         sS = sS.symmetrise('antipodal');
% %         sigma = stressTensor.uniaxial(loadAxis);
% %         % rotate the stress tensor into crystal coordinates
% %         sigmaLocal = inv(cells(ph).meanOrientation) * sigma;
% %         SF = sS.SchmidFactor(sigmaLocal);
% %
% %         % take the maxium along the rows
% %         [SFMax,active] = max(SF,[],2);
% %
% %         % second slip system
% %         SFmod=SF;
% %         for i=1:length(SFMax)
% %             SFmod(i,active(i))=NaN;
% %         end
% %         [SFMax2,active2] = max(SFmod,[],2);
% %
% %         % third slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active2(i))=NaN;
% %         end
% %         [~,active3] = max(SFmod,[],2);
% %
% %         % 4th slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active3(i))=NaN;
% %         end
% %         [~,active4] = max(SFmod,[],2);
% %
% %         % 5th slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active4(i))=NaN;
% %         end
% %         [~,active5] = max(SFmod,[],2);
% %
% %         % 6th slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active5(i))=NaN;
% %         end
% %         [~,active6] = max(SFmod,[],2);
% %         %__________________________________________________________________________________________________
% %         %% create the Schmid factor plots
% %                 % figtemp=figure;
% %                 % plot(cells(ph).boundary);
% %                 % hold on
% %                 % plot(cells(ph),SFMax)
% %                 % mtexColorbar
% %                 % setColorRange([0.2,0.5]);
% %                 % colormap modblueyelloworange;
% %                 % mtexTitle('max. Schmid Factor')
% %                 % saveas(figtemp, ['SFs_' datname '.png']);
% %                 % close all
% %
% %         % take the active slip system and rotate it in specimen coordinates
% %         sSactive = cells(ph).meanOrientation .* sS(active);
% %
% %         figtemp=figure;
% %         plot(ebsd_smoothedFill(ph), kam_smoothed./degree);
% %         hold on
% %         plot(grains.boundary, 'lineColor', [0.65,0.65,0.65]);
% %         setColorRange([0,2.5])
% %          colormap(KAM_WhiteBlueHeatColorMap)
% %         mtexTitle('Slip Systems highest 6 SFs')
% %         hold on
% %
% %             % calc and plot the slip plane of the 6th  max. slip system
% %             sSactive6 = cells(ph).meanOrientation .* sS(active6);
% %             quiver(cells(ph),sSactive6.trace,'color',[0.3 0.3 0.3],'linewidth', 0.5, 'lineStyle', ':');
% %             % and the slip direction of the 6th  max. slip system
% %             %     quiver(cells(ph),sSactive6.b,'color',[0.3 0.3 0.3],'linewidth',  0.5, 'lineStyle', ':');
% %
% %             % calc and plot the slip plane of the 5th  max. slip system
% %             sSactive5 = cells(ph).meanOrientation .* sS(active5);
% %             quiver(cells(ph),sSactive5.trace,'color',[0.3 0.3 0.3],'linewidth',  0.5, 'lineStyle', ':');
% %             % and the slip direction of the 5th  max. slip system
% %             %     quiver(cells(ph),sSactive5.b,'color',[0.3 0.3 0.3],'linewidth',  0.5, 'lineStyle', ':');
% %
% %             % calc and plot the slip plane of the 4th  max. slip system
% %             sSactive4 = cells(ph).meanOrientation .* sS(active4);
% %             quiver(cells(ph),sSactive4.trace,'color',[0.5 0.5 0.3],'linewidth', 1);
% %             % and the slip direction of the 4th  max. slip system
% %             %     quiver(cells(ph),sSactive4.b,'color',[0.5 0.5 0.3],'linewidth', 1);
% %
% %             %the third  max. slip system
% %             sSactive3 = cells(ph).meanOrientation .* sS(active3);
% %             quiver(cells(ph),sSactive3.trace,'color',[0.5 0.5 1],'linewidth', 1);
% %             %     quiver(cells(ph),sSactive3.b,'color',[0.5 0.5 1],'linewidth', 1);
% %
% %             %the second max. slip system
% %             sSactive2 = cells(ph).meanOrientation .* sS(active2);
% %             quiver(cells(ph),sSactive2.trace,'color',[0.3,0.8,0.4],'linewidth', 1);
% %             %     quiver(cells(ph),sSactive2.b,'color',[0.3,0.8,0.4],'linewidth', 1);
% %
% %             % the maximum slip system
% %             quiver(cells(ph),sSactive.trace,'color',[0.9,0.2,0.4]);
% %             %     quiver(cells(ph),sSactive.b,'color',[0.9,0.2,0.4]);
% %
% %             %     legend('6th Highest SF sS', '','5th Highest SF sS', '','4th Highest SF sS', '', '3rd Highest SF sS', '','2nd Highest SF sS', '','Highest SF sS', '')
% %             legend('6th Highest SF sS', '5th Highest SF sS', '4th Highest SF sS',  '3rd Highest SF sS', '2nd Highest SF sS', 'Highest SF sS')
% %
% %             %plot(cells.boundary, 'color', 'k');
% %             hold off
% %             saveas(figtemp, ['6HighestSF-sS_' datname '.png']);
% %             end
% %         elseif SFinv==0
% %             sS = slipSystem.fcc(ebsd_smoothedFill(ph).CS);
% %         sS = sS.symmetrise('antipodal');
% %         sigma = stressTensor.uniaxial(loadAxis);
% %         % rotate the stress tensor into crystal coordinates
% %         sigmaLocal = inv(cells(ph).meanOrientation) * sigma;
% %         SF = sS.SchmidFactor(sigmaLocal);
% %
% %         % take the maxium along the rows
% %         [SFMax,active] = max(SF,[],2);
% %
% %         % second slip system
% %         SFmod=SF;
% %         for i=1:length(SFMax)
% %             SFmod(i,active(i))=NaN;
% %         end
% %         [SFMax2,active2] = max(SFmod,[],2);
% %
% %         % third slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active2(i))=NaN;
% %         end
% %         [~,active3] = max(SFmod,[],2);
% %
% %         % 4th slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active3(i))=NaN;
% %         end
% %         [~,active4] = max(SFmod,[],2);
% %
% %         % 5th slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active4(i))=NaN;
% %         end
% %         [~,active5] = max(SFmod,[],2);
% %
% %         % 6th slip system
% %         for i=1:length(SFMax)
% %             SFmod(i,active5(i))=NaN;
% %         end
% %         [~,active6] = max(SFmod,[],2);
% %         %__________________________________________________________________________________________________
% %         %% create the Schmid factor plots
% %                 % figtemp=figure;
% %                 % plot(cells(ph).boundary);
% %                 % hold on
% %                 % plot(cells(ph),SFMax)
% %                 % mtexColorbar
% %                 % setColorRange([0.2,0.5]);
% %                 % colormap modblueyelloworange;
% %                 % mtexTitle('max. Schmid Factor')
% %                 % saveas(figtemp, ['SFs_' datname '.png']);
% %                 % close all
% %
% %         % take the active slip system and rotate it in specimen coordinates
% %         sSactive = cells(ph).meanOrientation .* sS(active);
% %
% %         figtemp=figure;
% %         plot(ebsd_smoothedFill(ph), kam_smoothed./degree);
% %         hold on
% %         plot(grains.boundary, 'lineColor', [0.65,0.65,0.65]);
% %         setColorRange([0,5])
% %          colormap('bonemod')
% %         mtexTitle('Slip Systems highest 6 SFs')
% %         hold on
% %
% %             % calc and plot the slip plane of the 6th  max. slip system
% %             sSactive6 = cells(ph).meanOrientation .* sS(active6);
% %             quiver(cells(ph),sSactive6.trace,'color',[0.3 0.3 0.3],'linewidth', 0.5, 'lineStyle', ':');
% %             % and the slip direction of the 6th  max. slip system
% %             %     quiver(cells(ph),sSactive6.b,'color',[0.3 0.3 0.3],'linewidth',  0.5, 'lineStyle', ':');
% %
% %             % calc and plot the slip plane of the 5th  max. slip system
% %             sSactive5 = cells(ph).meanOrientation .* sS(active5);
% %             quiver(cells(ph),sSactive5.trace,'color',[0.3 0.3 0.3],'linewidth',  0.5, 'lineStyle', ':');
% %             % and the slip direction of the 5th  max. slip system
% %             %     quiver(cells(ph),sSactive5.b,'color',[0.3 0.3 0.3],'linewidth',  0.5, 'lineStyle', ':');
% %
% %             % calc and plot the slip plane of the 4th  max. slip system
% %             sSactive4 = cells(ph).meanOrientation .* sS(active4);
% %             quiver(cells(ph),sSactive4.trace,'color',[0.5 0.5 0.3],'linewidth', 1);
% %             % and the slip direction of the 4th  max. slip system
% %             %     quiver(cells(ph),sSactive4.b,'color',[0.5 0.5 0.3],'linewidth', 1);
% %
% %             %the third  max. slip system
% %             sSactive3 = cells(ph).meanOrientation .* sS(active3);
% %             quiver(cells(ph),sSactive3.trace,'color',[0.5 0.5 1],'linewidth', 1);
% %             %     quiver(cells(ph),sSactive3.b,'color',[0.5 0.5 1],'linewidth', 1);
% %
% %             %the second max. slip system
% %             sSactive2 = cells(ph).meanOrientation .* sS(active2);
% %             quiver(cells(ph),sSactive2.trace,'color',[0.3,0.8,0.4],'linewidth', 1);
% %             %     quiver(cells(ph),sSactive2.b,'color',[0.3,0.8,0.4],'linewidth', 1);
% %
% %             % the maximum slip system
% %             quiver(cells(ph),sSactive.trace,'color',[0.9,0.2,0.4]);
% %             %     quiver(cells(ph),sSactive.b,'color',[0.9,0.2,0.4]);
% %
% %             %     legend('6th Highest SF sS', '','5th Highest SF sS', '','4th Highest SF sS', '', '3rd Highest SF sS', '','2nd Highest SF sS', '','Highest SF sS', '')
% %             legend('6th Highest SF sS', '5th Highest SF sS', '4th Highest SF sS',  '3rd Highest SF sS', '2nd Highest SF sS', 'Highest SF sS')
% %
% %             %plot(cells.boundary, 'color', 'k');
% %             hold off
% %             saveas(figtemp, ['6HighestSF-sS_' datname '.png']);
% % end
% %
% %
% %
end