function [RXedGrainFraction, RXedGrains] = RxxAnalysis(n,foldername, datname,ebsd_smoothedFill, filename, ebsd1)
%% grain average KAM
ph='Aluminium';
% ebsd_goodR_ungrid = ebsd_goodR;
close all

% ebsd_goodR_ungrid(1) = [];

disp 'calc grains smoothed'
ebsd_smoothedFill(isnan(ebsd_smoothedFill.x))=[];
[grains_smoothed,ebsd_smoothedFill.grainId]=calcGrains(ebsd_smoothedFill,'alpha',2.0 , 'angle',[3, 1].*degree);
grains_smoothed=smooth(grains_smoothed,4);
%grains_smoothed('n')=[];
disp 'calc gKAM smoothed'
[gkam, gid, ebsd_smoothedFill.prop.KAM] = gKAM(ebsd_smoothedFill,'threshold',5*degree,'order',2);


%% grain average BC_______ Achtung: Bei automatischer Backgroundcorrection BandContrast Auswertung nicht ganz korrekt
% [gBC_smoothed,gFe, ~] = gBC(ebsd_smoothedFill);
[gBC_smoothed, ~, ~ , gid] = gBC(ebsd_smoothedFill);

% grainBC=figure;
% plot(grains_smoothed(gid_smoothed),gBC_smoothed)
% mtexColorMap black2white
% mtexTitle('Grain av. BC smoothed')
% mtexColorMap black2white
% mtexColorbar
% saveas(grainBC,['grainBC_' datname '.png']);
% [gBCraw, gidraw] = gBC(ebsd_goodR_ungrid);

% highGBCsmoothed=sum(grains_smoothed(gid_smoothed(gBC_smoothed>(0.70*max(gBC_smoothed)))).numPixel)/sum(grains_smoothed.numPixel)*100;

%% Grain orientation spread, Grain average KAM, Grain average BC and Rxed parts of them
GOSfig=figure;
plot(grains_smoothed('a'),grains_smoothed('a').GOS./degree)
setColorRange([0 12])
mtexTitle('Grain Orientation Spread smoothed')
mtexColorbar('GOS, degree');
saveas(GOSfig,['GOS_' datname '.png']);

lowGKAM=mean(sum(grains_smoothed(gid(gkam<0.55*degree)).numPixel)/sum(grains_smoothed(grains_smoothed('A')).numPixel)*100)
lowKAM=mean(sum(ebsd_smoothedFill.prop.KAM<0.55*degree)/length(ebsd_smoothedFill)*100)
highBC=mean(sum(ebsd_smoothedFill(ph).bc>(0.7*max(ebsd_smoothedFill.bc)))/length(ebsd_smoothedFill(ph))*100)

% highBCgrainofAll=sum(grainsraw_smoothed(gidraw(gBCraw>(0.9*max(gBCraw)))).numPixel)/length(ebsd_goodR_ungrid)*100

%% combined query
fe = exist('ebsd_smoothedFill.prop.Fe_Ka1', 'var');
if fe ==1
RXedGrains=grains_smoothed(grains_smoothed.GOS<1.15*degree & gBC_smoothed>(0.55*max(gBC_smoothed)) & gFe<2.2*nanmean(gFe) & gkam<0.7*degree);
% für CR
% RXedGrains=grains_smoothed(grains_smoothed.GOS<1*degree & gBC_smoothed>(0.8*max(gBC_smoothed)) & gFe<nanmean(gFe))
% für part RX
% RXedGrains=grains_smoothed(grains_smoothed.GOS<1.45*degree & gkam<(0.6*degree) & gFe<2.2*nanmean(gFe)) %gBC_smoothed>(0.50*max(gBC_smoothed)) & gFe<1.15*nanmean(gFe))
try
 RXedGrains(gFe>nanmean(gFe))=[];
end
else
 RXedGrains=grains_smoothed(grains_smoothed.GOS<1.25*degree & gBC_smoothed>(0.7*max(gBC_smoothed)) & gkam<0.55*degree);
end

RXedGrains=(RXedGrains(ph));
RXedGrains=RXedGrains(RXedGrains.equivalentRadius>0.4);

% RXedGrains=grains_smoothed(gkam<0.5*degree & gBC_smoothed>(0.6*max(gBC_smoothed)))
% RXedGrains=grainsraw_smoothed(grainsraw_smoothed.GOS<1.35*degree & gBCraw>(0.75*max(gBCraw)));
% notRXedGrains=grains_smoothed(gkam>0.505*degree & gBC_smoothed<(0.499*max(gBC_smoothed)))

RXedGrainFraction=sum(RXedGrains(ph).numPixel)/sum(grains_smoothed(ph).numPixel)*100

%%
newMtexFigure
plot(ebsd1, ebsd1.bc, "figSize","large")%, 'micronbar', 'off');
setColorRange([55,242]);
mtexColorMap black2white

hold on
plot(grains_smoothed.innerBoundary,'linecolor','r','linewidth',0.85)
hold on
plot(RXedGrains, RXedGrains, 'FaceColor','GoldenRod', 'FaceAlpha', 0.45, 'LineColor', 'none');

hold on
plot(grains_smoothed.boundary, 'LineColor', 'DarkBlue', 'LineWidth', 1.3);


% plot(notRXedGrains, 'Color', 'r', 'FaceAlpha', 0.4)

% legend(['RX grains'])
% legend('off')
mtexTitle ('Rx Grains yellow')
saveFigure(['RXgrains_' datname '.png']);
close all

%%
writecell({'RXedGrainFraction';'highBC';'lowGKAM';'lowKAM'}, filename, 'Sheet', datname, 'Range', 'A28:A31');
writematrix([RXedGrainFraction; highBC; lowGKAM; lowKAM], filename, 'Sheet', datname, 'Range', 'B28:B31');

docXls = [foldername 'Documentation.xlsx'];
writecell({'RXedGrainFraction';'highBC';'lowGKAM';'lowKAM'}, docXls, 'Sheet', 1, 'Range', 'AN1:AQ1');
writematrix([RXedGrainFraction; highBC; lowGKAM; lowKAM], docXls, 'Sheet', 1, 'Range', ['AN' num2str(n+1) ':AQ' num2str(n+1)]);

binwidthGOS=0.25;
binedgesGOS=0:binwidthGOS:15;
histAreaWeighGOS = histogram(grains_smoothed, grains_smoothed.GOS./degree, binedgesGOS);
sheetGOSHist = "GOS hist, area weighted";

colLet = excelColumn(n+1);
% Header (dataset name) in row 1
writecell({char(datname)}, docXls, "Sheet", sheetGOSHist, "Range", sprintf("%s1", colLet));
writecell({"BinCenters"}, docXls, "Sheet", sheetGOSHist, "Range", "A2");
writecell({"GOS hist, area weighted"}, docXls, "Sheet", sheetGOSHist, "Range", "A1");
writecell(num2cell(binedgesGOS(1:60)+binwidthGOS/2).', docXls, 'Sheet', sheetGOSHist, 'Range', 'A3');
writematrix([histAreaWeighGOS.Values(:)], docXls, 'Sheet', sheetGOSHist, 'Range', sprintf("%s3", colLet));
%%
binwidthGkam=0.2;
binedgesGKAM=0:binwidthGkam:5;
histAreaWeighGKAM = histogram(grains_smoothed, gkam./degree, binedgesGKAM);
sheetGKAMHist = "gKAM - area weighted";

colLet = excelColumn(n+1);
% Header (dataset name) in row 1
writecell({char(datname)}, docXls, "Sheet", sheetGKAMHist, "Range", sprintf("%s1", colLet));
writecell({"BinCenters"}, docXls, "Sheet", sheetGKAMHist, "Range", "A2");
writecell({"GOS hist, area weighted"}, docXls, "Sheet", sheetGKAMHist, "Range", "A1");
writecell(num2cell(binedgesGOS(1:25)+binwidthGOS/2).', docXls, 'Sheet', sheetGKAMHist, 'Range', 'A3');
writematrix([histAreaWeighGKAM.Values(:)], docXls, 'Sheet', sheetGKAMHist, 'Range', sprintf("%s3", colLet));

 %% ICOTOM grain size texture
% %
% CS=ebsd_smoothedFill(ph).CS
% 
% sym_Bs=unique((specimenSymmetry('mmm') * orientation.byMiller([0 1 1],[2 1 1],CS))); %(ph))));
% sym_S=unique((specimenSymmetry('mmm') * orientation.byMiller([1 2 3],[6 3 4],CS)));
% sym_Cu=unique((specimenSymmetry('mmm') * orientation.byMiller([1 1 2],[1 1 1],CS)));
% sym_Goss=unique((specimenSymmetry('mmm') * orientation.byMiller([0 1 1],[1 0 0],CS)));
% sym_rotGoss=unique((specimenSymmetry('mmm') * orientation.byMiller([0 -1 1],[0 1 1],CS)));
% sym_CND45=unique((specimenSymmetry('mmm') *orientation.byMiller([1 0 0],[0 1 1],CS)));
% sym_C=unique((specimenSymmetry('mmm') * orientation.byMiller([0 0 1],[1 0 0],CS)));
% sym_Q=unique((specimenSymmetry('mmm') *orientation.byMiller([0 1 3],[2 -3 1],CS)));
% sym_P=unique((specimenSymmetry('mmm') * orientation.byMiller([0 1 1],[1 2 2],CS)));
% sym_shear2=unique((specimenSymmetry('mmm') * orientation.byMiller([1 1 1],[1 1 2],CS)));
% sym_CND22=unique((specimenSymmetry('mmm') * orientation.byMiller([0 0 1],[3 1 0],CS))); % auch [0 1 3],[1 0 0] ... Abw. 3.365°
% sym_R=unique((specimenSymmetry('mmm') * orientation.byMiller([1 2 4],[2 1 1],CS))); %4-fold
% sym_CRD10=unique((specimenSymmetry('mmm') * orientation.byEuler([0, 10, 0]*degree,CS))); % auch [0 1 3],[1 0 0] ... Abw. 3.365°
% sym_CND10=unique((specimenSymmetry('mmm') * orientation.byEuler([10,0, 0]*degree,CS))); % auch [0 1 3],[1 0 0] ... Abw. 3.365°
% 
% 
% % Grain size development
% 
% GrainTXTAngle=12*degree;
% pxnumber=length(ebsd_smoothedFill('Aluminium').orientations);
% % 9components
% Bsgrains=findByOrientation(RXedGrains, sym_Bs, GrainTXTAngle);
% Cugrains=findByOrientation(RXedGrains, sym_Cu, GrainTXTAngle);
% Cgrains=findByOrientation(RXedGrains, sym_C, GrainTXTAngle);
% Sgrains=findByOrientation(RXedGrains, sym_S, GrainTXTAngle);
% Rgrains=findByOrientation(RXedGrains, sym_R, GrainTXTAngle);
% Gossgrains=findByOrientation(RXedGrains, sym_Goss, GrainTXTAngle);
% CND22grains=findByOrientation(RXedGrains, sym_CND22, GrainTXTAngle);
% CND45grains=findByOrientation(RXedGrains, sym_CND45, GrainTXTAngle);
% Pgrains=findByOrientation(RXedGrains, sym_P, GrainTXTAngle);
% Qgrains=findByOrientation(RXedGrains, sym_Q, GrainTXTAngle);
% 
% %
% Bs_ECD=mean(Bsgrains.equivalentRadius*2);
% Bs_n=length(Bsgrains);
% Bs_area=length(ebsd_smoothedFill(Bsgrains))/pxnumber*100;
% 
% S_ECD=mean(Sgrains.equivalentRadius*2);
% S_n=length(Sgrains);
% S_area=length(ebsd_smoothedFill(Sgrains))/pxnumber*100;
% 
% Cu_ECD=mean(Cugrains.equivalentRadius*2);
% Cu_n=length(Cugrains);
% Cu_area=length(ebsd_smoothedFill(Cugrains))/pxnumber*100;
% 
% Goss_ECD=mean(Gossgrains.equivalentRadius*2);
% Goss_n=length(Gossgrains);
% Goss_area=length(ebsd_smoothedFill(Gossgrains))/pxnumber*100;
% 
% CND45_ECD=mean(CND45grains.equivalentRadius*2);
% CND45_n=length(CND45grains);
% CND45_area=length(ebsd_smoothedFill(CND45grains))/pxnumber*100;
% 
% CND22_ECD=mean(CND22grains.equivalentRadius*2);
% CND22_n=length(CND22grains);
% CND22_area=length(ebsd_smoothedFill(CND22grains))/pxnumber*100;
% 
% C_ECD=mean(Cgrains.equivalentRadius*2);
% C_n=length(Cgrains);
% C_area=length(ebsd_smoothedFill(Cgrains))/pxnumber*100;
% 
% Q_ECD=mean(Qgrains.equivalentRadius*2);
% Q_n=length(Qgrains);
% Q_area=length(ebsd_smoothedFill(Qgrains))/pxnumber*100;
% 
% P_ECD=mean(Pgrains.equivalentRadius*2);
% P_n=length(Pgrains);
% P_area=length(ebsd_smoothedFill(Pgrains))/pxnumber*100;
% 
% R_ECD=mean(Rgrains.equivalentRadius*2);
% R_n=length(Rgrains);
% R_area=length(ebsd_smoothedFill(Rgrains))/pxnumber*100;
% 
% ECDsOri= [{Bs_ECD}, { S_ECD}, {Cu_ECD}, {Goss_ECD}, {CND45_ECD}, {CND22_ECD}, {C_ECD}, {Q_ECD}, {P_ECD},{R_ECD}];
% NumberGrainsOri= [{Bs_n}, {S_n}, {Cu_n}, {Goss_n}, {CND45_n}, {CND22_n}, {C_n}, {Q_n}, {P_n},{R_n}];
% Area= [{Bs_area}, {S_area}, {Cu_area}, {Goss_area}, {CND45_area}, {CND22_area}, {C_area}, {Q_area}, {P_ECD},{R_area}];
% 
% TxCompheader={'Brass','S','Copper','Goss','Rot. Cube ND45' , 'C-ND22', 'Cube', 'Q Lage','P Lage','R'};
% xlswrite([foldername 'Documentation_GSize.xlsx'],{datname},['A' num2str(n+2) ':A' num2str(n+2)]);
% xlswrite([foldername 'Documentation_GSize.xlsx'],TxCompheader,'B1:K1');
% xlswrite([foldername 'Documentation_GSize.xlsx'],{'ECDs'},'B2:B2');
% 
% xlswrite([foldername 'Documentation_GSize.xlsx'],TxCompheader,'L1:U1');
% xlswrite([foldername 'Documentation_GSize.xlsx'],{'Number Grains Ori'},'L2:L2');
% 
% xlswrite([foldername 'Documentation_GSize.xlsx'],TxCompheader,'V1:AE1');
% xlswrite([foldername 'Documentation_GSize.xlsx'],{'Area fraction of total'},'V2:V2');
% 
% 
% xlswrite([foldername 'Documentation_GSize.xlsx'],ECDsOri,['B' num2str(n+2) ':K' num2str(n+2)]);
% xlswrite([foldername 'Documentation_GSize.xlsx'],NumberGrainsOri,['L' num2str(n+2) ':U' num2str(n+2)]);
% xlswrite([foldername 'Documentation_GSize.xlsx'],Area,['V' num2str(n+2) ':AE' num2str(n+2)]);
% xlswrite([foldername 'Documentation_GSize.xlsx'],{'RXedGrainFraction'},'AF1:AF1');
% xlswrite([foldername 'Documentation_GSize.xlsx'],RXedGrainFraction, ['AF' num2str(n+2) ':AF' num2str(n+2)]);
% xlswrite([foldername 'Documentation_GSize.xlsx'],{'number nuclei'},['AG1:AG1']);
% xlswrite([foldername 'Documentation_GSize.xlsx'],length(RXedGrains),['AG' num2str(n+2) ':AG' num2str(n+2)]);
% xlswrite([foldername 'Documentation_GSize.xlsx'],{'avg ECD all'},['AH1:AH1']);
% xlswrite([foldername 'Documentation_GSize.xlsx'],mean(RXedGrains.equivalentRadius*2),['AH' num2str(n+2) ':AH' num2str(n+2)]);

 
 
 
 
 %% find tilt 40° boundaries
% rxGbs=RXedGrains.boundary(ph,ph)
% oriGB=ebsd_smoothedFill('id', rxGbs.ebsdId).orientations
% axS = axis(oriGB(:,1),oriGB(:,2),'antipodal')
% 
% figure
% plot(ebsd1, ebsd1.bc, 'micronbar', 'off');
% setColorRange([55,242]);
% colormap gray
% freezeColors
% hold on
% plot(RXedGrains.boundary,'linewidth',2)
% plot(rxGbs,angle(rxGbs.direction,axS)./degree,'linewidth',1)
% mtexColorMap blue2red
% 
% setColorRange([0 90])
% mtexColorbar

%

%% Für paper Flo
% GBMiso=grains_smoothed.boundary('a','a');
% 
% RX_GBmapMiso=figure;
% plot(ebsd_smoothedFill, ebsd_smoothedFill.bc, 'MicronBar', 'off', 'figSize', 'large');
% setColorRange('tight');
% colormap b2wcontrast
% freezeColors
% hold on
% plot(RXedGrains, RXedGrains, 'FaceColor','LightCoral', 'FaceAlpha', 0.35, 'LineColor', 'none');
% % plot(RXedGrains, RXedGrains, 'FaceColor','LightCoral', 'FaceAlpha', 0.35, 'LineColor', 'none', 'micronbar', 'off');
% freezeColors
% hold on
% plot(grains_smoothed.boundary, 'linewidth',1.5, 'lineColor', 'DarkBlue');
% colormap parula
% set(gca, 'Color', [0.15 0.15 0.15]);
% set(gcf, 'InvertHardcopy', 'off')
% 
% saveas(RX_GBmapMiso, ['RxedGrains_logColorCodedGBAngle5to63deg_' datname '.png']);
% saveas(RX_GBmapMiso, ['RxedGrains_logColorCodedGBAngle5to63deg_' datname '.fig']);

