function  [odfCombined ,VolTexture_odfCombined,VolTextureEBSDsm_fill, drehung] = TextureAnalysis_mtex6(particleRem, TXTtolAngle,ebsd1plot,h,datname, ebsd_smoothedFill,autoc, grains, foldername, n, rot, ph)

%% definition of symmetric equivalents
% if particleRem==1
% ph='Aluminium'
% else
% ph=CS{2}.mineral
% end
docXls = fullfile(foldername, 'Documentation.xlsx');
CS=ebsd_smoothedFill(ph).CS

% in Miller Indizes; J. Hirsch ist eigentlich für Euler Winkel
sym_Bs=unique((specimenSymmetry('mmm') * orientation.byMiller([0 1 1],[2 1 1],CS)));  %(ph))));
sym_S=unique((specimenSymmetry('mmm') * orientation.byMiller([1 2 3],[6 3 4],CS)));
sym_Cu=unique((specimenSymmetry('mmm') * orientation.byMiller([1 1 2],[1 1 1],CS)));
sym_Goss=unique((specimenSymmetry('mmm') * orientation.byMiller([0 1 1],[1 0 0],CS)));
sym_rotGoss=unique((specimenSymmetry('mmm') * orientation.byMiller([0 -1 1],[0 1 1],CS)));
sym_CND45=unique((specimenSymmetry('mmm') *orientation.byMiller([1 0 0],[0 1 1],CS)));  %PSN oder Raabe und Lücke und andere sagen, das kann auch von Shear kommen.
sym_C=unique((specimenSymmetry('mmm') * orientation.byMiller([0 0 1],[1 0 0],CS)));
sym_Q=unique((specimenSymmetry('mmm') *orientation.byMiller([0 1 3],[2 -3 1],CS)));
sym_P=unique((specimenSymmetry('mmm') * orientation.byMiller([0 1 1],[1 2 2],CS)));
sym_shear2=unique((specimenSymmetry('mmm') * orientation.byMiller([1 1 1],[1 1 2],CS)));
sym_CND22=unique((specimenSymmetry('mmm') * orientation.byMiller([0 0 1],[3 1 0],CS))); % auch [0 1 3],[1 0 0] ... Abw. 3.365°
sym_R=unique((specimenSymmetry('mmm') * orientation.byMiller([1 2 4],[2 1 1],CS))); %4-fold
sym_CRD10=unique((specimenSymmetry('mmm') * orientation.byEuler([0, 10, 0]*degree,CS))); % auch [0 1 3],[1 0 0] ... Abw. 3.365°
sym_CND10=unique((specimenSymmetry('mmm') *  orientation.byEuler([10,0, 0]*degree,CS))); % auch [0 1 3],[1 0 0] ... Abw. 3.365°

% oriR=ebsd_smoothedFill(ph).orientations;
oriR=ebsd_smoothedFill(ph).orientations;

%% Kernel calculation and ODF estimation
% grP = grains(ph);
% Ng  = numel(grP);
%
% if Ng > 4000
% idx = randperm(Ng, 4000);
% grainsSam = grP(idx);  % still uses subSet internally, but far less often than discreteSample
% else
% grainsSam = grP;
% end
%
% weights=grainsSam.meanOrientation;
% Wn = norm(weights(:));
% ind = ~isnan(Wn);  %weights must be normalized
%
% psiGrains=calcKernel(weights);
%
% psiMod=calcKernelMod(oriR,'method','magicRule'); %modifizierte Magic Rule: kappa, numOri und halfwidth können/sollen beeinflusst werden
% odf_kernelMod=calcDensity(oriR,'kernel',psiGrains);


% % fix, for plotting, only intermediately detailed ODF
% psi = SO3DeLaValleePoussinKernel('halfwidth',7.5*degree);
% odf_kernelMod=calcDensity(oriR,'kernel',psi);
% odf_grainKernel=calcDensity(weights, 'weights', Wn(ind));

% odfCombined = 0.5* odf_grainKernel +0.5*odf_kernelMod

% Paper GHADIR 1.5 Mn
odfCombined=calcDensity(oriR, 'halfwidth', 5*degree)
% odfCombined = odf_kernelMod

if length(grains(ph))>400
    if autoc==1
        try
            % wenn auto-zentriert werden soll
            odftemp=calcDensity(oriR, 'halfwidth', 7.5*degree);
            [~,drehungtemp]=centerSpecimen(odftemp)
            % end
            % if the total suggested rotation for centering is > 7°, it is usually wrong
            % -> then no centering is applied
            drehungtemp.angle./degree

            if drehungtemp.angle < 7*degree
                drehung=drehungtemp
                oriR=drehung*oriR;
                ebsd_smoothedFill(ph).orientations=drehung*ebsd_smoothedFill(ph).orientations;
                odfCombined= drehung*odfCombined;
            else
                odfCombined= odfCombined;
                drehung=rotation.byEuler([0,0,0])
            end
        end
    else
        odfCombined= odfCombined;
        drehung=rotation.byEuler([0,0,0])
    end
else
    odfCombined= odfCombined;
    drehung=rotation.byEuler([0,0,0])
end
if exist("drehung")==0;
    drehung=rotation.byEuler([0,0,0]);
end

odfCombined;
Entr=entropy(odfCombined);
tIndex=norm(odfCombined)^2;


writecell({'TextureIndex','Entropy'}, docXls, 'Range', 'X1:Y1');
writematrix([tIndex, Entr], docXls, 'Range', sprintf('X%d:Y%d', n+1, n+1));

%%
% figtemp=figure;
% plotPDF(odfCombined, h,'contourf', 'xAxisdirection', 'north', 'minmax');
%  setColorRange([0,5])
% mtexColorMap LaboTeX
% mtexColorbar;
% saveas(figtemp, ['ODF_combined_' datname '.png']);
plotx2north
figtemp=figure;
plotPDF(oriR, h,'Points', 2500, 'MarkerSize', 1.7, 'minmax', 'grid', 'on','xAxisDirection', 'north');
saveas(figtemp, ['PFscattered_' datname '.png']);

try
    figtemp=figure;
    odf_sy=odfCombined;
    odf_sy.SS=specimenSymmetry('mmm');
    % plot(odf_sy, 'phi2',[0,45,65]*degree, 'contourf')
    plot(odf_sy, 'phi2',[0,45,65]*degree, 'contourf', 'figSize' , 'small')
    setColorRange([0,7])

    mtexColorbar;
    colormap(WhiteBlueHeatColorMap)
    % aM=[0:0.6/63:0.6]
    % set(gca, 'Alphamap', aM)
    % setColorRange([0,3])

    saveas(figtemp, ['ODFsymmSect_combined_' datname '.png']);
    % auch als matlab figure, weil es lang plottet, dann kann man es einfacher
    % modifizieren
    % saveas(figtemp, ['ODFsymmSect_combined_' datname '.fig']);
end

% close all
% figtemp=figure;
% plotIPDF(odfCombined, zvector,'figSize' , 'tiny' )
% colormap(WhiteBlueHeatColorMap)
% setColorRange([0 7])
% mtexColorbar;
% mtexTitle('  ')
% saveFigure(['IPF_z_' datname '.png'])
plotx2east
%%
% save([foldername 'ODF' datname '.mat'],'odfCombined' ,'-v7.3')


%% volume Fraction Determination EBSD
volEBSD_B = volume(oriR, sym_Bs(1), TXTtolAngle)*100+volume(oriR, sym_Bs(2), TXTtolAngle)*100;
volEBSD_S = volume(oriR, sym_S(1), TXTtolAngle)*100+volume(oriR, sym_S(2), TXTtolAngle)*100+volume(oriR, sym_S(3), TXTtolAngle)*100+volume(oriR, sym_S(4), TXTtolAngle)*100;
volEBSD_Cu = volume(oriR, sym_Cu(1), TXTtolAngle)*100+volume(oriR, sym_Cu(2), TXTtolAngle)*100;
volEBSD_Goss = volume(oriR, sym_Goss(1), TXTtolAngle)*100;
volEBSD_invGoss = volume(oriR, sym_rotGoss(1), TXTtolAngle)*100;
volEBSD_C = volume(oriR, sym_C(1), TXTtolAngle)*100;
volEBSD_rotC = volume(oriR, sym_CND45(1), TXTtolAngle)*100;
volEBSD_Q = volume(oriR, sym_Q(1), TXTtolAngle)*100+volume(oriR, sym_Q(2), TXTtolAngle)*100+volume(oriR, sym_Q(3), TXTtolAngle)*100+volume(oriR, sym_Q(4), TXTtolAngle)*100;
volEBSD_P = volume(oriR, sym_P(1), TXTtolAngle)*100+volume(oriR, sym_P(2), TXTtolAngle)*100;
volEBSD_CRD10 = volume(oriR, sym_CRD10(1), TXTtolAngle)*100+volume(oriR, sym_CRD10(2), TXTtolAngle)*100;
volEBSD_CND10 = volume(oriR, sym_CND10(1), TXTtolAngle)*100+volume(oriR, sym_CND10(2), TXTtolAngle)*100;
volEBSD_CND22 = volume(oriR, sym_CND22(1), TXTtolAngle)*100+volume(oriR, sym_CND22(2), TXTtolAngle)*100;
volEBSD_R = volume(oriR, sym_R(1), TXTtolAngle)*100+volume(oriR, sym_R(2), TXTtolAngle)*100+volume(oriR, sym_R(3), TXTtolAngle)*100+volume(oriR, sym_R(4), TXTtolAngle)*100;
volEBSD_shear2 = volume(oriR, sym_shear2(1), TXTtolAngle)*100+volume(oriR, sym_shear2(2), TXTtolAngle)*100;

%% volume Fraction Determination ODF
volODF_B = volume(odfCombined, sym_Bs(1), TXTtolAngle)*100+volume(odfCombined, sym_Bs(2), TXTtolAngle)*100;
volODF_S = volume(odfCombined, sym_S(1), TXTtolAngle)*100+volume(odfCombined, sym_S(2), TXTtolAngle)*100+volume(odfCombined, sym_S(3), TXTtolAngle)*100+volume(odfCombined, sym_S(4), TXTtolAngle)*100;
volODF_Cu = volume(odfCombined, sym_Cu(1), TXTtolAngle)*100+volume(odfCombined, sym_Cu(2), TXTtolAngle)*100;
volODF_Goss = volume(odfCombined, sym_Goss(1), TXTtolAngle)*100;
volODF_invGoss = volume(odfCombined, sym_rotGoss(1), TXTtolAngle)*100;
volODF_C = volume(odfCombined, sym_C(1), TXTtolAngle)*100;
volODF_rotC = volume(odfCombined, sym_CND45(1), TXTtolAngle)*100;
volODF_Q = volume(odfCombined, sym_Q(1), TXTtolAngle)*100+volume(odfCombined, sym_Q(2), TXTtolAngle)*100+volume(odfCombined, sym_Q(3), TXTtolAngle)*100+volume(odfCombined, sym_Q(4), TXTtolAngle)*100;
volODF_P = volume(odfCombined, sym_P(1), TXTtolAngle)*100+volume(odfCombined, sym_P(2), TXTtolAngle)*100;
volODF_CRD10 = volume(odfCombined, sym_CRD10(1), TXTtolAngle)*100+volume(odfCombined, sym_CRD10(2), TXTtolAngle)*100;
volODF_CND10 = volume(odfCombined, sym_CND10(1), TXTtolAngle)*100+volume(odfCombined, sym_CND10(2), TXTtolAngle)*100;
volODF_CND22 = volume(odfCombined, sym_CND22(1), TXTtolAngle)*100+volume(odfCombined, sym_CND22(2), TXTtolAngle)*100;
volODF_R = volume(odfCombined, sym_R(1), TXTtolAngle)*100+volume(odfCombined, sym_R(2), TXTtolAngle)*100+volume(odfCombined, sym_R(3), TXTtolAngle)*100+volume(odfCombined, sym_R(4), TXTtolAngle)*100;
volODF_shear2 = volume(odfCombined, sym_shear2(1), TXTtolAngle)*100+volume(odfCombined, sym_shear2(2), TXTtolAngle)*100;

% %% relative vol. fractions EBSD
% % volume Fraction Determination EBSD
% relvolEBSD_B = 1/2/0.6759*(volume(oriR, sym_Bs(1), TXTtolAngle)*100+volume(oriR, sym_Bs(2), TXTtolAngle)*100);
% relvolEBSD_S =  1/4/0.6759*(volume(oriR, sym_S(1), TXTtolAngle)*100+volume(oriR, sym_S(2), TXTtolAngle)*100+volume(oriR, sym_S(3), TXTtolAngle)*100+volume(oriR, sym_S(4), TXTtolAngle)*100);
% relvolEBSD_Cu = 1/2/0.6759*( volume(oriR, sym_Cu(1), TXTtolAngle)*100+volume(oriR, sym_Cu(2), TXTtolAngle)*100);
% relvolEBSD_Goss =  1/0.6759*(volume(oriR, sym_Goss(1), TXTtolAngle)*100);
% relvolEBSD_invGoss =  1/0.6759*(volume(oriR, sym_rotGoss(1), TXTtolAngle)*100);
% relvolEBSD_C =  1/0.6759*(volume(oriR, sym_C(1), TXTtolAngle)*100);
% relvolEBSD_rotC =  1/0.6759*(volume(oriR, sym_CND45(1), TXTtolAngle)*100);
% relvolEBSD_Q =  1/4/0.6759*(volume(oriR, sym_Q(1), TXTtolAngle)*100+volume(oriR, sym_Q(2), TXTtolAngle)*100+volume(oriR, sym_Q(3), TXTtolAngle)*100+volume(oriR, sym_Q(4), TXTtolAngle)*100);
% relvolEBSD_P =  1/2/0.6759*(volume(oriR, sym_P(1), TXTtolAngle)*100+volume(oriR, sym_P(2), TXTtolAngle)*100);
% relvolEBSD_CRD10 =  1/2/0.6759*(volume(oriR, sym_CRD10(1), TXTtolAngle)*100+volume(oriR, sym_CRD10(2), TXTtolAngle)*100);
% relvolEBSD_CND10 =  1/2/0.6759*(volume(oriR, sym_CND10(1), TXTtolAngle)*100+volume(oriR, sym_CND10(2), TXTtolAngle)*100);
% relvolEBSD_CND22 =  1/2/0.6759*(volume(oriR, sym_CND22(1), TXTtolAngle)*100+volume(oriR, sym_CND22(2), TXTtolAngle)*100);
% relvolEBSD_R =  1/4/0.6759*(volume(oriR, sym_R(1), TXTtolAngle)*100+volume(oriR, sym_R(2), TXTtolAngle)*100+volume(oriR, sym_R(3), TXTtolAngle)*100+volume(oriR, sym_R(4), TXTtolAngle)*100);
% relvolEBSD_shear2 =  1/2/0.6759*(volume(oriR, sym_shear2(1), TXTtolAngle)*100+volume(oriR, sym_shear2(2), TXTtolAngle)*100);
%%
namesVolTextureodfCombined =categorical({'Brass', 'S' , 'Copper', 'Goss','Rot. Cube ND45','Rot. Cube ND22', 'Rot. Cube RD10',...
    'Rot. Cube ND10', 'Cube' , 'Q', 'P', 'R','shear 2', 'Inv. Goss'});
namesVolTextureodfCombined = reordercats(namesVolTextureodfCombined,string(namesVolTextureodfCombined));

VolTexture_odfCombined =[sum(volODF_B), sum(volODF_S), sum(volODF_Cu), sum(volODF_Goss), sum(volODF_rotC), sum(volODF_CND22),  sum(volODF_CRD10),sum(volODF_CND10),sum(volODF_C), sum(volODF_Q), sum(volODF_P), sum(volODF_R), sum(volODF_shear2), sum(volODF_invGoss)];
VolTextureEBSDsm_fill =[volEBSD_B, volEBSD_S, volEBSD_Cu, volEBSD_Goss, volEBSD_rotC, volEBSD_CND22, volEBSD_CRD10, volEBSD_CND10, volEBSD_C, volEBSD_Q, volEBSD_P,  volEBSD_R, volEBSD_shear2, volEBSD_invGoss];

% Summe der Abweichungen
% abweichung = sum(abs(VolTexture_odfCombined-VolTextureEBSDsm_fill));

BarCompVolNew =figure;
bar(namesVolTextureodfCombined,[VolTexture_odfCombined.', VolTextureEBSDsm_fill.'], 'grouped');
legend('ODF comb.', 'EBSD ori.');%, 'ODF grain Kernel' , 'ODF Kernel magig rule mod');

if max(VolTexture_odfCombined,VolTextureEBSDsm_fill)<16
    ylim([0,16]);
    set(gca,'YGrid','on','YMinorGrid','on','YTick',[2 4 6 8 10 12 14 16]);
else
    set(gca,'YGrid','on','YMinorGrid','on')%,'YTick',[2 4 6 8 10 12 14 16]);
end

xlabel('Main Texture Components', 'FontWeight', 'bold');
ylabel('Volume Fraction (%)', 'FontWeight', 'bold');
saveas(BarCompVolNew, ['VolumeFractions_odfCombined_' datname '.png']);
% %%
% relVolTextureEBSDsm_fill =[relvolEBSD_B, relvolEBSD_S, relvolEBSD_Cu, relvolEBSD_Goss, relvolEBSD_rotC,  relvolEBSD_CND22, relvolEBSD_CRD10, relvolEBSD_CND10, relvolEBSD_C, relvolEBSD_Q, relvolEBSD_P, relvolEBSD_R, relvolEBSD_shear2, relvolEBSD_invGoss];
% relBarCompVolNew =figure;
% bar(namesVolTextureodfCombined,relVolTextureEBSDsm_fill.');
% legend('EBSD id. TXT *mrd');%, 'ODF grain Kernel' , 'ODF Kernel magig rule mod');
% saveas(relBarCompVolNew, ['relVolumeFractions_odfCombined_' datname '.png']);

%% Add additional information to the info-sheet

TxCompheader={'Brass','S','Copper','Goss','Rot. Cube ND45' , 'C-ND22', 'C-RD10','C-ND10', 'Cube',  'Q Lage','P Lage','R', 'shear 2','Rot. Goss'};
filename = ['doc_' datname '.xlsx'];

writecell(TxCompheader, filename,'Sheet', datname, 'Range', 'B19:O19');

writecell({'ODF data'}, filename, 'Sheet', datname, 'Range', 'A20:A20');

writematrix(VolTexture_odfCombined, filename,  'Sheet', datname, 'Range', 'B20:O20');

writecell({'EBSD data'}, filename, 'Sheet', datname, 'Range', 'A21:A21');

writematrix(VolTextureEBSDsm_fill, filename, 'Sheet', datname, 'Range', 'B21:O21');


writecell(TxCompheader, docXls, ...
    'Sheet', 1, 'Range', 'Z1:AM1');

writematrix(VolTextureEBSDsm_fill, docXls, ...
    'Sheet', 1, 'Range', ['Z' num2str(n+1) ':AM' num2str(n+1)]);


writecell({'relative Intensity from direct EBSD data'}, filename, ...
    'Sheet', datname, 'Range', 'A23:A23');



TxAdditional={'Drehung Axis -x','Drehung Axis -y','Drehung Axis -z','Drehung Angle', ...
    'number Orientations','psi Mod. Bandwidth','psi Mod. Halfwidth', ...
    'mean Grain orientations','psi Grains. Bandwidth', ...
    'psi Grains. Halfwidth','TextureIndex','Entropy'};

writecell(TxAdditional, filename, ...
    'Sheet', datname, 'Range', 'A16:L16');

writecell([{num2str(drehung.axis.x)},  {num2str(drehung.axis.y)}, ...
    {num2str(drehung.axis.z)}, {num2str(drehung.angle/degree)}, ...
    {num2str(length(oriR))}, {'-'},{'-'},{'-'},{'-'},{'-'}, ...
    {norm(odfCombined)^2}, {num2str(Entr)}], ...
    filename, 'Sheet', datname, 'Range', 'A17:L17');
%% plot in AxisAngle Space
% figure;
% plot(odfCombined, 'AxisAngle')
% ylim([0 45])
% xlim([0 45])
% setColorRange([0 15])
% % colormap(b2rI)
% set(gca, 'CameraViewAngle', 11.2);
% colorbar
% colormap(WhiteBlueHeatColorMap)
% aM=[0:0.6/63:0.6];
% set(gca, 'Alphamap', aM)
% saveFigure([foldername datname '_RodriguesODF_blueRed.png'])
% % saveFigure([foldername datname '_RodriguesODF_blueRed.fig'])


%% TexturMaps
% %  updated colormaps
% %  Colorize according to the ideal orientations (paper)
%
% ipfKeyT = spotColorKey(CS);%{1,2});
% ipfKeyT.center = [sym_Bs(1),sym_Bs(2),...
%   sym_S(1), sym_S(2), sym_S(3), sym_S(4),...
%   sym_Cu(1),sym_Cu(2),...
%   sym_Goss(1),...
%   sym_C, ...
%   sym_CND45,...
%   sym_CND22(1), sym_CND22(2), ...
%   sym_CRD10(1),sym_CRD10(2),...
%   sym_P(1), sym_P(2)...
%   ];  %workaround for
% % equal array dimensions, as it only colors, double orientations should not cause troubles
%
% ipfKeyT.color = [[0,0.3,1];[0,0.3,1];...
%   [0,1,0.8];[0,1,0.8];[0,1,0.8];[0,1,0.8];...
%   [1,0,1];[1,0,1];...
%   [0.3 1 0.0];...
%   [1,0,0];...
%   [0.8,1,0];...
%   [1,0.5,0];[1,0.5,0];...
%   [0.4,0,1];[0.4,0,1];...
%   [0.5 0.7 1];[0.5 0.7 1]...
%   ];
%
%
% ipfKeyT.psi = deLaValleePoussinKernel('halfwidth',0.72*TXTtolAngle); % 10° Abweichung erlaubt
%
% strNames = {'\color[rgb]{0,0.3,1} Brass',...
%   '\color[rgb]{0,1,0.8} S',...
%   '\color[rgb]{1,0,1} Copper', ...
%   '\color[rgb]{0.2,1,0} Goss',...
%   '\color[rgb]{1,0,0} Cube',...
%   '\color[rgb]{0.8, 0.8, 0} CubeND45',...
%   '\color[rgb]{1,0.5,0} CubeND22', ...
%   '\color[rgb]{0.4,0,1} CubeRD10 ',...
%   '\color[rgb]{0.5 0.7 1} P'...
%   };
%
% % create a figure colored by the texture components (+-10°)
% newMtexFigure;
%
% colors=ipfKeyT.orientation2color(ebsd_smoothedFill(ph).orientations);
% plot(ebsd_smoothedFill(ph),colors)
% % hold on
% % plot(grains.boundary, 'lineWidth', 0.5, 'lineColor', [0.2,0.2,0.2]);
% set(gca, 'Color','k')
% freezeColors
% hold on
% plot(ebsd1plot(ph), ebsd1plot(ph).bc, 'FaceAlpha',0.45)
% plot(ebsd1plot(~ebsd1plot.isIndexed), ebsd1plot(~ebsd1plot.isIndexed).bc, 'FaceAlpha',0.85)
% % plot(ebsd1plot(~ebsd1.isIndexed), ebsd1plot(~ebsd1.isIndexed).bc, 'FaceAlpha',0.75)
%
% colormap gray
% % saveFigure(['TextureComponents_' datname '.png']);
%
% annotation('textbox', 'String', strNames, 'BackgroundColor',[1,1,1], 'Position',[-0, 0, 0.4, 0.4],'FitBoxToText','on');
% % hold off
% % saveFigure(['TextureComponentslegend_' datname '.png']);
% % hold on
%
% %%  Additional: create a figure colored by the texture components + unit cells
%
% cS = crystalShape.cube(CS);
% isBig = grains(ph).grainSize>180;
% backrot=inv(rotation.byEuler([rot(n,3) rot(n,2) rot(n,1)]*degree, CS));
% CrystalOrientationRotGrains=grains(ph);
% CrystalOrientationRotGrains.meanOrientation=backrot*CrystalOrientationRotGrains.meanOrientation;
% plot(CrystalOrientationRotGrains(isBig),0.7*cS, 'FaceAlpha', 0.5)
% hold off
% saveFigure(['TextureComponentsCrystal_' datname '.png']);

%% TexturMaps
%  updated colormaps
%  Colorize according to the ideal orientations (paper)

ipfKeyT = spotColorKey(CS);%{1,2});
ipfKeyT.center = [sym_Bs(1),sym_Bs(2),...
    sym_S(1), sym_S(2), sym_S(3), sym_S(4),...
    sym_Cu(1),sym_Cu(2),...
    sym_Goss(1),...
    sym_C, ...
    sym_CND45,...
    sym_CND22(1), sym_CND22(2), ...
    sym_P(1), sym_P(2)...
    ];  %workaround for
% equal array dimensions, as it only colors, double orientations should not cause troubles

ipfKeyT.color = [[0,0.3,1];[0,0.3,1];...
    [0,1,0.8];[0,1,0.8];[0,1,0.8];[0,1,0.8];...
    [1,0,1];[1,0,1];...
    [0.3 1 0.0];...
    [1,0,0];...
    [0.8,1,0];...
    [1,0.5,0];[1,0.5,0];...
    [0.5 0.7 1];[0.5 0.7 1]...
    ];


ipfKeyT.psi = SO3DeLaValleePoussinKernel('halfwidth',1*TXTtolAngle); % 10° Abweichung erlaubt

strNames = {'\color[rgb]{0,0.3,1} Brass',...
    '\color[rgb]{0,1,0.8} S',...
    '\color[rgb]{1,0,1} Copper', ...
    '\color[rgb]{0.2,1,0} Goss',...
    '\color[rgb]{1,0,0} Cube',...
    '\color[rgb]{0.8, 0.8, 0} CubeND45',...
    '\color[rgb]{1,0.5,0} CubeND22', ...
    '\color[rgb]{0.5 0.7 1} P'...
    };

% create a figure colored by the texture components (+-10°)
newMtexFigure;

colors=ipfKeyT.orientation2color(ebsd_smoothedFill(ph).orientations);
plot(ebsd_smoothedFill(ph),colors)
hold on
plot(grains.boundary, 'lineWidth', 0.5, 'lineColor', [0.2,0.2,0.2]);
% set(gca, 'Color','k')
freezeColors
hold on
plot(ebsd1plot(ph), ebsd1plot(ph).bc, 'FaceAlpha',0.35)
plot(ebsd1plot(~ebsd1plot.isIndexed), ebsd1plot(~ebsd1plot.isIndexed).bc)%, 'FaceAlpha',0.85)
% plot(ebsd1plot(~ebsd1plot.isIndexed), ebsd1plot(~ebsd1plot.isIndexed).bc, 'FaceAlpha',0.75)

colormap gray
% saveFigure(['TextureComponents_' datname '.png']);

annotation('textbox', 'String', strNames, 'BackgroundColor',[1,1,1], 'Position',[-0, 0, 0.4, 0.4],'FitBoxToText','on');
hold off
saveFigure(['TextureComponentslegend_' datname '.png']);
% hold on

%%  Additional: create a figure colored by the texture components + unit cells
% !! TO PATCH !! %
% cS = crystalShape.cube(CS);
% isBig = grains(ph).numPixel>280;
%
% backrot=inv(rotation.byEuler([rot(n,3) rot(n,2) rot(n,1)]*degree, CS));
% % backrot=inv(rotation.byEuler([rotInitial(3) rotInitial(2) rotInitial(1)]*degree, CS));
% CrystalOrientationRotGrains=grains(ph);
% CrystalOrientationRotGrains.meanOrientation=backrot*CrystalOrientationRotGrains.meanOrientation;
% hold on
% plot(CrystalOrientationRotGrains(isBig),0.7*cS,'FaceColor', 'LightGray', 'FaceAlpha', 0.35)
% hold off
% saveFigure(['TextureComponentsCrystal_' datname '.png']);

%%
% How much of orientations has the 111 plane in the surface plane?

% Define the Miller indices for the {111} plane
miller111 = Miller(1,1,1, ebsd_smoothedFill('A').orientations.CS);

% Define the z-direction in the sample coordinate system
zDirection = vector3d.Z;

% Calculate the angle between the {111} direction and the sample z-direction
angleToZ = angle(ebsd_smoothedFill('A').orientations * miller111, zDirection);

% Set a tolerance angle (in radians, e.g., 5 degrees)
tolerance = 15* degree;

% Find the orientations where the angle is within the tolerance
alignedOrientations = angleToZ < tolerance;

% Calculate the volume fraction of aligned orientations
volumeFraction = sum(alignedOrientations) / numel(alignedOrientations)* 100;

% Display the result as a percentage
fprintf('Volume fraction with {111} plane in surface plane: %.2f%%\n', volumeFraction );

writecell({'pct of 111 plane in surface'}, ...
    docXls, 'Sheet', 1, 'Range', 'AT1:AT1');

writematrix(volumeFraction, ...
    docXls, 'Sheet', 1, ...
    'Range', ['AT' num2str(n+1) ':AT' num2str(n+1)]);

% How much of orientations has the 100 plane in the surface plane?

% Define the Miller indices for the {100} plane
miller100 = Miller(1,0,0, ebsd_smoothedFill('A').orientations.CS);

% Define the z-direction in the sample coordinate system
zDirection = vector3d.Z;

% Calculate the angle between the {111} direction and the sample z-direction
angleToZ = angle(ebsd_smoothedFill('A').orientations * miller100, zDirection);


% Find the orientations where the angle is within the tolerance
alignedOrientations = angleToZ < tolerance;

% Calculate the volume fraction of aligned orientations
volumeFraction = sum(alignedOrientations) / numel(alignedOrientations)* 100;

% Display the result as a percentage
fprintf('Volume fraction with {100} plane in surface plane: %.2f%%\n', volumeFraction );


writecell({'pct of 100 plane in surface'}, ...
    docXls, 'Sheet', 1, 'Range', 'AU1:AU1');

writematrix(volumeFraction, docXls, 'Sheet', 1, 'Range', ['AU' num2str(n+1) ':AU' num2str(n+1)]);

% How much of orientations has the 110 plane in the surface plane?

% Define the Miller indices for the {110} plane
miller110 = Miller(1,1,0, ebsd_smoothedFill('A').orientations.CS);

% Define the z-direction in the sample coordinate system
zDirection = vector3d.Z;

% Calculate the angle between the {110} direction and the sample z-direction
angleToZ = angle(ebsd_smoothedFill('A').orientations * miller110, zDirection);

% Find the orientations where the angle is within the tolerance
alignedOrientations = angleToZ < tolerance;

% Calculate the volume fraction of aligned orientations
volumeFraction = sum(alignedOrientations) / numel(alignedOrientations)* 100;

% Display the result as a percentage
fprintf('Volume fraction with {110} plane in surface plane: %.2f%%\n', volumeFraction );

writecell({'pct of 110 plane in surface'}, docXls, ...
    'Sheet', 1, 'Range', 'AV1:AV1');

writematrix(volumeFraction, docXls, ...
    'Sheet', 1, 'Range', ['AV' num2str(n+1) ':AV' num2str(n+1)]);





%%
% % Band analysis ACHTUNG Toleranz 15°
%
% grains15=calcGrains(ebsd_smoothedFill,'angle', 15*degree)
%
% grains15=grains15(ph);
% cubeGrains=findByOrientation(grains15, sym_C, 15*degree);
% maxDimXCube=zeros(length(cubeGrains),1);
% maxDimYCube=zeros(length(cubeGrains),1);
% numGrains=length(cubeGrains);
%
% for i=1:numGrains
% maxDimXCube(i)=max(cubeGrains(i).x)-min(cubeGrains(i).x);
% maxDimYCube(i)=max(cubeGrains(i).y)-min(cubeGrains(i).y);
%
% end
%
% avg_maxDimXCube=mean(maxDimXCube);
% std_maxDimXCube=std(maxDimXCube);
% avg_maxDimYCube=mean(maxDimYCube);
% std_maxDimYCube=std(maxDimYCube);
%
% SGrains=findByOrientation(grains15, sym_S, 15*degree);
% maxDimXS=zeros(length(SGrains),1);
% maxDimYS=zeros(length(SGrains),1);
% numGrains=length(SGrains);

% for i=1:numGrains
% maxDimXS(i)=max(SGrains(i).x)-min(SGrains(i).x);
% maxDimYS(i)=max(SGrains(i).y)-min(SGrains(i).y);
%
% end
%
% avg_maxDimXS=mean(maxDimXS);
% std_maxDimXS=std(maxDimXS);
% avg_maxDimYS=mean(maxDimYS);
% std_maxDimYS=std(maxDimYS);
%
% GossGrains=findByOrientation(grains15, sym_Goss, 15*degree);
% maxDimXGoss=zeros(length(GossGrains),1);
% maxDimYGoss=zeros(length(GossGrains),1);
% numGrains=length(GossGrains);
%
% for i=1:numGrains
% maxDimXGoss(i)=max(GossGrains(i).x)-min(GossGrains(i).x);
% maxDimYGoss(i)=max(GossGrains(i).y)-min(GossGrains(i).y);
%
% end
%
% avg_maxDimXGoss=mean(maxDimXGoss);
% std_maxDimXGoss=std(maxDimXGoss);
% avg_maxDimYGoss=mean(maxDimYGoss);
% std_maxDimYGoss=std(maxDimYGoss);
%
% CND22Grains=findByOrientation(grains15, sym_CND22, 15*degree);
% maxDimXCND22=zeros(length(CND22Grains),1);
% maxDimYCND22=zeros(length(CND22Grains),1);
% numGrains=length(CND22Grains);
%
% for i=1:numGrains
% maxDimXCND22(i)=max(CND22Grains(i).x)-min(CND22Grains(i).x);
% maxDimYCND22(i)=max(CND22Grains(i).y)-min(CND22Grains(i).y);
%
% end
%
% avg_maxDimXCND22=mean(maxDimXCND22);
% std_maxDimXCND22=std(maxDimXCND22);
% avg_maxDimYCND22=mean(maxDimYCND22);
% std_maxDimYCND22=std(maxDimYCND22);
%
% avg_maxDimXGoss=mean(maxDimXGoss);
% std_maxDimXGoss=std(maxDimXGoss);
% avg_maxDimYGoss=mean(maxDimYGoss);
% std_maxDimYGoss=std(maxDimYGoss);
%
% % all grains
% maxDimXall=zeros(length(grains15),1);
% maxDimYall=zeros(length(grains15),1);
% numGrains=length(grains15);
%
% for i=1:numGrains
% maxDimXall(i)=max(grains15(i).x)-min(grains15(i).x);
% maxDimYall(i)=max(grains15(i).y)-min(grains15(i).y);
%
% end
%
% avg_maxDimXall=mean(maxDimXall);
% std_maxDimXall=std(maxDimXall);
% avg_maxDimYall=mean(maxDimYall);
% std_maxDimYall=std(maxDimYall);
%
%
% avg_maxDimX_All=[avg_maxDimXCube, avg_maxDimXS, avg_maxDimXGoss, avg_maxDimXCND22,avg_maxDimXall];
% std_maxDimX_All=[std_maxDimXCube, std_maxDimXS, std_maxDimXGoss, std_maxDimXCND22,std_maxDimXall];
% avg_maxDimY_All=[avg_maxDimYCube, avg_maxDimYS, avg_maxDimYGoss, avg_maxDimYCND22,avg_maxDimYall];
% std_maxDimY_All=[std_maxDimYCube, std_maxDimYS, std_maxDimYGoss, std_maxDimYCND22,std_maxDimYall];
%
%
% xlswrite(filename,{'Cube Grains', 'S Grains','Goss Grains','all Grains'}, datname,'B25:E25');
% xlswrite(filename,{'max. Dim. X', '+/- X','max. Dim. Y','+/- Y'}.', datname,'A26:A29');
% xlswrite(filename,avg_maxDimX_All, datname,'B26:E26');
% xlswrite(filename,std_maxDimX_All, datname,'B27:E27');
% xlswrite(filename,avg_maxDimY_All, datname,'B28:E28');
% xlswrite(filename,std_maxDimY_All, datname,'B29:E29');
%
% figtemp=figure;
% b=bar(categorical({'Cube', 'S', 'Goss','CubeND22', 'all grains'}),[avg_maxDimX_All;avg_maxDimY_All],'grouped');
% hold on
% % Get the x coordinate of the bars
% x = nan(2, 5);
% for i = 1:2
% x(i,:) = b(i).XEndPoints;
% end
% % Plot the errorbars
% errorbar(x,[avg_maxDimX_All;avg_maxDimY_All],[std_maxDimX_All;std_maxDimY_All],'k','linestyle','none');
% hold off
% legend('max. grain length in RD', 'max. grains length in ND');
% saveas(figtemp, ['Banding of txt components 15degTolerance_' datname '.png']);

%% Seperate by GOS
VolTextureEBSDsm_fill =[VolTextureEBSDsm_fill(1:7) VolTextureEBSDsm_fill(9:14)];
[grains, ebsd_smoothedFill.grainId]=calcGrains(ebsd_smoothedFill, "alpha",2, "angle", 6*degree)
GOS = ebsd_smoothedFill.grainMean(calcGROD(ebsd_smoothedFill, grains('A')).angle./degree, grains('A'));
SubStrGrains=(grains(GOS>1));
RXgrains=(grains(GOS<1));

ebsdRX=ebsd_smoothedFill(RXgrains);
ebsdSubstr=ebsd_smoothedFill(SubStrGrains);

oriR=ebsdRX.orientations;
volEBSD_B = volume(oriR, sym_Bs(1), TXTtolAngle)*100+volume(oriR, sym_Bs(2), TXTtolAngle)*100;
volEBSD_S = volume(oriR, sym_S(1), TXTtolAngle)*100+volume(oriR, sym_S(2), TXTtolAngle)*100+volume(oriR, sym_S(3), TXTtolAngle)*100+volume(oriR, sym_S(4), TXTtolAngle)*100;
volEBSD_Cu = volume(oriR, sym_Cu(1), TXTtolAngle)*100+volume(oriR, sym_Cu(2), TXTtolAngle)*100;
volEBSD_Goss = volume(oriR, sym_Goss(1), TXTtolAngle)*100;
volEBSD_invGoss = volume(oriR, sym_rotGoss(1), TXTtolAngle)*100;
volEBSD_C = volume(oriR, sym_C(1), TXTtolAngle)*100;
volEBSD_rotC = volume(oriR, sym_CND45(1), TXTtolAngle)*100;
volEBSD_Q = volume(oriR, sym_Q(1), TXTtolAngle)*100+volume(oriR, sym_Q(2), TXTtolAngle)*100+volume(oriR, sym_Q(3), TXTtolAngle)*100+volume(oriR, sym_Q(4), TXTtolAngle)*100;
volEBSD_P = volume(oriR, sym_P(1), TXTtolAngle)*100+volume(oriR, sym_P(2), TXTtolAngle)*100;
volEBSD_CRD10 = volume(oriR, sym_CRD10(1), TXTtolAngle)*100+volume(oriR, sym_CRD10(2), TXTtolAngle)*100;
volEBSD_CND22 = volume(oriR, sym_CND22(1), TXTtolAngle)*100+volume(oriR, sym_CND22(2), TXTtolAngle)*100;
volEBSD_R = volume(oriR, sym_R(1), TXTtolAngle)*100+volume(oriR, sym_R(2), TXTtolAngle)*100+volume(oriR, sym_R(3), TXTtolAngle)*100+volume(oriR, sym_R(4), TXTtolAngle)*100;
volEBSD_shear2 = volume(oriR, sym_shear2(1), TXTtolAngle)*100+volume(oriR, sym_shear2(2), TXTtolAngle)*100;
VolebsdRX =[volEBSD_B, volEBSD_S, volEBSD_Cu, volEBSD_Goss, volEBSD_rotC, volEBSD_CND22, volEBSD_CRD10, volEBSD_C, volEBSD_Q, volEBSD_P,  volEBSD_R, volEBSD_shear2, volEBSD_invGoss];

oriR=ebsdSubstr.orientations;
volEBSD_B = volume(oriR, sym_Bs(1), TXTtolAngle)*100+volume(oriR, sym_Bs(2), TXTtolAngle)*100;
volEBSD_S = volume(oriR, sym_S(1), TXTtolAngle)*100+volume(oriR, sym_S(2), TXTtolAngle)*100+volume(oriR, sym_S(3), TXTtolAngle)*100+volume(oriR, sym_S(4), TXTtolAngle)*100;
volEBSD_Cu = volume(oriR, sym_Cu(1), TXTtolAngle)*100+volume(oriR, sym_Cu(2), TXTtolAngle)*100;
volEBSD_Goss = volume(oriR, sym_Goss(1), TXTtolAngle)*100;
volEBSD_invGoss = volume(oriR, sym_rotGoss(1), TXTtolAngle)*100;
volEBSD_C = volume(oriR, sym_C(1), TXTtolAngle)*100;
volEBSD_rotC = volume(oriR, sym_CND45(1), TXTtolAngle)*100;
volEBSD_Q = volume(oriR, sym_Q(1), TXTtolAngle)*100+volume(oriR, sym_Q(2), TXTtolAngle)*100+volume(oriR, sym_Q(3), TXTtolAngle)*100+volume(oriR, sym_Q(4), TXTtolAngle)*100;
volEBSD_P = volume(oriR, sym_P(1), TXTtolAngle)*100+volume(oriR, sym_P(2), TXTtolAngle)*100;
volEBSD_CRD10 = volume(oriR, sym_CRD10(1), TXTtolAngle)*100+volume(oriR, sym_CRD10(2), TXTtolAngle)*100;
volEBSD_CND22 = volume(oriR, sym_CND22(1), TXTtolAngle)*100+volume(oriR, sym_CND22(2), TXTtolAngle)*100;
volEBSD_R = volume(oriR, sym_R(1), TXTtolAngle)*100+volume(oriR, sym_R(2), TXTtolAngle)*100+volume(oriR, sym_R(3), TXTtolAngle)*100+volume(oriR, sym_R(4), TXTtolAngle)*100;
volEBSD_shear2 = volume(oriR, sym_shear2(1), TXTtolAngle)*100+volume(oriR, sym_shear2(2), TXTtolAngle)*100;
VolebsdSubStr =[volEBSD_B, volEBSD_S, volEBSD_Cu, volEBSD_Goss, volEBSD_rotC, volEBSD_CND22, volEBSD_CRD10, volEBSD_C, volEBSD_Q, volEBSD_P,  volEBSD_R, volEBSD_shear2, volEBSD_invGoss];

namesVolTextureodfCombined =categorical({'Brass', 'S' , 'Copper', 'Goss','Rot. Cube ND45','Rot. Cube ND22', 'Rot. Cube RD10',...
    'Cube' , 'Q', 'P', 'R','shear 2', 'Inv. Goss'});
namesVolTextureodfCombined = reordercats(namesVolTextureodfCombined,string(namesVolTextureodfCombined));


close all
newMtexFigure
bar(namesVolTextureodfCombined,[VolebsdRX.', VolebsdSubStr.', VolTextureEBSDsm_fill.'], 'grouped');
legend('lowGOS Grains', 'HighGOS Grains', 'All Grains');
ylabel('Volume Fraction (%)', 'FontWeight', 'bold');
nextAxis
plot(RXgrains, 'FaceColor', 'DarkMagenta','FaceAlpha', 0.7, 'DisplayName', 'lowGOS Grains');
hold on;
plot(SubStrGrains, 'FaceColor', 'DarkGoldenrod','FaceAlpha', 0.7, 'DisplayName', 'HighGOS Grains');
hold on;
plot(grains.boundary);
hold on
plot(grains.innerBoundary, 'lineColor','Gray');
saveFigure(['largeSmallGrainsVsTexture' datname '.png']);
TxCompheader={'Brass','S','Copper','Goss','Rot. Cube ND45' , 'C-ND22', 'C-RD10', 'Cube',  'Q Lage','P Lage','R', 'shear 2','Rot. Goss'};

sheetName = 'SmallVSLargeGrain';

writecell({'all Grains'},  docXls, 'Sheet', sheetName, 'Range', 'B1:B1');
writecell({'Substructure Grains'}, docXls, 'Sheet', sheetName, 'Range', 'O1:O1');
writecell({'RX Grains'}, docXls, 'Sheet', sheetName, 'Range', 'AB1:AB1');

writecell({datname}, docXls, 'Sheet', sheetName, 'Range', ['A' num2str(n+2) ':A' num2str(n+2)]);

% header row (B2:AN2) — must be cell
writecell([TxCompheader, TxCompheader, TxCompheader], docXls, 'Sheet', sheetName, 'Range', 'B2:AN2');

% data row (B(n+2):AN(n+2)) — numeric row vector
writematrix([VolTextureEBSDsm_fill, VolebsdSubStr, VolebsdRX], docXls, 'Sheet', sheetName, 'Range', ['B' num2str(n+2) ':AN' num2str(n+2)]);

%%  Grain size development
% GrainTXTAngle=12*degree;
%
% [grains, ebsd_smoothedFill.grainId]= calcGrains(ebsd_smoothedFill,'alpha',2.0,'angle',gbThreshold);
%
% % 9components
% Bsgrains=findByOrientation(grains, sym_Bs, GrainTXTAngle);
% Cugrains=findByOrientation(grains, sym_Cu, GrainTXTAngle);
% Cgrains=findByOrientation(grains, sym_C, GrainTXTAngle);
% Sgrains=findByOrientation(grains, sym_S, GrainTXTAngle);
% Rgrains=findByOrientation(grains, sym_R, GrainTXTAngle);
% Gossgrains=findByOrientation(grains, sym_Goss, GrainTXTAngle);
% CND22grains=findByOrientation(grains, sym_CND22, GrainTXTAngle);
% CND45grains=findByOrientation(grains, sym_CND45, GrainTXTAngle);
% Pgrains=findByOrientation(grains, sym_P, GrainTXTAngle);
% Qgrains=findByOrientation(grains, sym_Q, GrainTXTAngle);
%
% %%
% Bs_ECD=mean(Bsgrains.equivalentRadius*2);
% Bs_n=length(Bsgrains);
% Bs_area=sum((Bsgrains.equivalentRadius).^2*pi);
%
% S_ECD=mean(Sgrains.equivalentRadius*2);
% S_n=length(Sgrains);
% S_area=sum((Sgrains.equivalentRadius).^2*pi);
%
% Cu_ECD=mean(Cugrains.equivalentRadius*2);
% Cu_n=length(Cugrains);
% Cu_area=sum((Cugrains.equivalentRadius).^2*pi);
%
% Goss_ECD=mean(Gossgrains.equivalentRadius*2);
% Goss_n=length(Gossgrains);
% Goss_area=sum((Gossgrains.equivalentRadius).^2*pi);
%
% CND45_ECD=mean(CND45grains.equivalentRadius*2);
% CND45_n=length(CND45grains);
% CND45_area=sum((CND45grains.equivalentRadius).^2*pi);
%
% CND22_ECD=mean(CND22grains.equivalentRadius*2);
% CND22_n=length(CND22grains);
% CND22_area=sum((CND22grains.equivalentRadius).^2*pi);
%
% C_ECD=mean(Cgrains.equivalentRadius*2);
% C_n=length(Cgrains);
% C_area=sum((Cgrains.equivalentRadius).^2*pi);
%
% Q_ECD=mean(Qgrains.equivalentRadius*2);
% Q_n=length(Qgrains);
% Q_area=sum((Qgrains.equivalentRadius).^2*pi);
%
% P_ECD=mean(Pgrains.equivalentRadius*2);
% P_n=length(Pgrains);
% P_area=sum((Pgrains.equivalentRadius).^2*pi);
%
% R_ECD=mean(Rgrains.equivalentRadius*2);
% R_n=length(Rgrains);
% R_area=sum((Rgrains.equivalentRadius).^2*pi);
%
% ECDsOri= [{Bs_ECD}, { S_ECD}, {Cu_ECD}, {Goss_ECD}, {CND45_ECD}, {CND22_ECD}, {C_ECD}, {Q_ECD}, {P_ECD},{R_ECD}];
% NumberGrainsOri=  [{Bs_n}, {S_n}, {Cu_n}, {Goss_n}, {CND45_n}, {CND22_n}, {C_n}, {Q_n}, {P_n},{R_n}];
% Area=  [{Bs_area},  {S_area}, {Cu_area}, {Goss_area}, {CND45_area}, {CND22_area}, {C_area}, {Q_area}, {P_ECD},{R_area}];
%
% TxCompheader={'Brass','S','Copper','Goss','Rot. Cube ND45' , 'C-ND22', 'Cube',  'Q Lage','P Lage','R'};
% xlswrite([foldername 'Documentation_GSize.xlsx'],TxCompheader,'B1:K1');
%  xlswrite([foldername 'Documentation_GSize.xlsx'],{'ECDs'},'B2:K2');
%
% xlswrite([foldername 'Documentation_GSize.xlsx'],TxCompheader,'L1:U1');
%  xlswrite([foldername 'Documentation_GSize.xlsx'],{'Number Grains Ori'},'L2:U2');
%
% xlswrite([foldername 'Documentation_GSize.xlsx'],TxCompheader,'V1:AE1');
%  xlswrite([foldername 'Documentation_GSize.xlsx'],{'Areas'},'V2:AE2');
%
%
% xlswrite([foldername 'Documentation_GSize.xlsx'],ECDsOri,['B' num2str(n+2) ':K' num2str(n+2)]);
% xlswrite([foldername 'Documentation_GSize.xlsx'],NumberGrainsOri,['L' num2str(n+2) ':U' num2str(n+2)]);
% xlswrite([foldername 'Documentation_GSize.xlsx'],Area,['V' num2str(n+2) ':AE' num2str(n+2)]);
%
%
%
%
%

end
