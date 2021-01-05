% % % zero crossing testing  (find zero upward, copy fs 4000, find next zero upward.
% clear all, clc, tic, clf;
% n=16000
% t=linspace(0,2*pi,n);
% y=cos (6*t)+sin(4*t);
% 
% find_zero = diff(sign(y));
% indx_up = find(find_zero>0); %find all upward going zeros
% indx_down = find(find_zero<0); %find all downward going zeros
% new_y=[];
% 
% fs_range_wanted=indx_up(1,1)+4000; %starts from first zero adds sample size wanted
% new_y=[y(indx_up(1,1):fs_range_wanted)]; %may have to minus 1
% ii=0;
% while (find_zero(1,fs_range_wanted+ii)  ~= 2);  %do while not going dwn and append 
%     ii=ii+1
%     y_pt_loc=fs_range_wanted+ii %what is the location of the point
%     new_y = [new_y, y(1,fs_range_wanted+ii)]; %append points
% end
% 
% 
% subplot(3,1,1);plot(y);title('Original Signal')
% subplot(3,1,2);plot(new_y);title('New signal')
% subplot(3,1,3);plot(find_zero);title('Zeros-Pos-Neg')

%%
% %Extract MFCC from Frequency-Domain Audio
% % Read in an audio file and convert it to a frequency representation.

% [audioIn,fs] = audioread('A2.wav');
% 
% win = hann(1024,'periodic');
% % stft=win;
% S = stft(audioIn,'Window',win,'OverlapLength',512,'Centered',false);
% 
% % To extract the mel-frequency cepstral coefficients, call mfcc with the frequency-domain audio. Ignore the log-energy.
% coeffs = mfcc(S,fs,'LogEnergy','Ignore');
% % In many applications, MFCC observations are converted to summary statistics for use in classification tasks. 
% %Plot probability density functions of each of the mel-frequency cepstral coefficients to observe their distributions.
% nbins = 60;
% for i = 1:size(coeffs,2)
%     figure
%     histogram(coeffs(:,i),nbins,'Normalization','pdf')
%     title(sprintf('Coefficient %d',i-1))
% end
%%
% %%Features Used for Classification
% [audioIn, fs] = audioread('Audio2 700KB.mp3');
% twoStart = 110e3;
% twoStop = 135e3;
% audioIn = audioIn(twoStart:twoStop);
% timeVector = linspace((twoStart/fs),(twoStop/fs),numel(audioIn));
% 
% figure
% plot(timeVector,audioIn)
% axis([(twoStart/fs) (twoStop/fs) -1 1])
% ylabel('Amplitude')
% xlabel('Time (s)')
% title('Utterance - Two')
% sound(audioIn,fs)
%%
% %Voice Activity Detection
% clc; clear all; close all;
% 
% [x,fs]=audioread('A2.wav');
% %% preprocessing-----------
% x=resample(x,8000,fs);
% fs=8000;
% x=x(:,1);  
% 
% x=x./max(abs(x)); %------- normalization-----------
% x=x-mean(x);  %------mean subtration----------
% figure;plot(x);ylabel('Normalized voice magnitude');xlabel('Sample no');grid on;axis tight;
% x=awgn(x,30,'measured'); 
% %% Mannual reference markings
% ref{1}=[1651 3500 5635 6850 9258 11090 14950 18570 21800 23600;... %first row starting points
%        3200 5400 6800 8770 10010 13820 16840 21300 22580 26650];   %second row ending points		
% 
% % ref{1}=[154	1880 4100; 1450	3540 4376];			
% % ref{2}=[1 2290 2650 3644; 2100 2480 3200 4376];
% % ref{3}=[177	1202 2757 4110; 1078 1885 3880 4376];
% 
% tmpp=ref{1};
% vseg=[];
% uvseg=1:length(x);
% 
% for kkk=1:size(tmpp,2)
%   vr=tmpp(1,kkk):tmpp(2,kkk);
%   vseg=[vseg vr];
%   uvseg(vr)=0;
% end
% uvseg(uvseg==0)=[];
% 
% %% Overlapped windowing
% 
% L = length(x);
% Bl = floor(0.03*fs); % 30 ms long window
% for i=L+1:L+1+Bl
%     x(i)=0;
% end
% 
% for i=1:L
%     Start = i;
%     End = i+Bl-1;
%     xB = x(Start:End); % current frame
%     
%     mag(i) = sum(abs(xB)); % feature extraction
% end
% 
% x=x(1:L);
% 
% %% threshold computation
%  n=fs*10e-3;
%  % mag_th= adaptive_th(mag,n);
% threshold=x;
% mag_th= threshold(mag);
% %% end point detection
% %
% 
% %% performance measure
% 
%  [Pcv1,Pcn1,Pc1,Pf1,S1,PP1]=performance_mesures(vseg,uvseg,vmag,uvmag);
% 
% 
% 
% %% Plot results
% figure;
% subplot(3,1,1);plot(x);hold on;plot(Gmag,'r');ylabel('speech');grid on;axis tight;
% subplot(3,1,2);plot(mag);hold on;plot(mag_th+mag.*0,'r');ylabel('AE');grid on;axis tight;
% subplot(3,1,3);plot(x.*Gmag');ylabel('VAD');xlabel('sample no');grid on;axis tight;
% hold on;scatter(SPmag,0.*SPmag);hold on;scatter(EPmag,0.*EPmag,'r');
%%
% clear
% close all
% clc
% [s1, fs] = audioread('A2.wav');
% s2 = audioread('A3.wav');
% ampMax = 0.5;
% 
% s1_output = audioNormalization_YW(s1, ampMax);
% s2_output = audioNormalization_YW(s2, ampMax);
% % sound(s1, fs);
% figure
% subplot(2,2,1)
% plot(s1)
% xlim([1 length(s1)])
% ylim([-1 1])
% title('original s1')
% subplot(2,2,2)
% plot(s2)
% xlim([1 length(s2)])
% ylim([-1 1])
% title('original s2')
% subplot(2,2,3)
% plot(s1_output)
% xlim([1 length(s1_output)])
% ylim([-1 1])
% title('normalized s1')
% subplot(2,2,4)
% plot(s2_output)
% xlim([1 length(s2_output)])
% ylim([-1 1])
% title('normalized s2')
%%
% [y, Fs] = audioread('A2.wav');
% histogram(y, 'FaceColor', 'red');
% figure(1),
% grid on;

% [w, Fs1] = audioread('A3.wav');
% histogram(w, 'FaceColor', 'blue');
% figure(1),
% grid on;
% 
% [v, Fs2] = audioread('A2.wav');
% histogram(w, 'FaceColor', 'cyan');
% figure(2),
% grid on;
%%
   % Algorithm for image validation
    % Open the two images which will be compared
    name2=input('Image name ( automated segmentation)     ','s');
    img_automated=imread(name2,'png');
    figure (1), imshow(img_automated), title('Image automated')
    name=input('Image name ( manual segmentation)     ','s');
    img_manual=imread(name,'png');
    img_manual_gray=rgb2gray(img_manual);
    figure (2), imshow (img_manual),title('Image manual')
    img_automated_gray=rgb2gray(img_automated);
    %img_double=im2double(img_automated_gray);
    figure (3), imshow (img_automated_gray), title (' Image converted to double ');
    imcontrast
    %uiwait(img_automated_gray)
    img_automated_eq=adapthisteq(img_automated_gray);
    figure (5), imshow (img_automated_eq), title (' Image after histogram equalization ');
    img_automated_gray=rgb2gray(img_automated);
    figure (6), imshowpair(img_manual,img_automated_eq)
    title('Images overlap')
    %Step 2: Choose Subregions of Each Image
    %It is important to choose regions that are similar.The image sub_automated
    %will be the template, and must be smaller than the image sub_manual. 
    % interactively
    [sub_manual,rect_manual] = imcrop(img_manual); % choose the pepper below the onion
    [sub_automated,rect_automated] = imcrop(img_automated_gray); % choose the whole onion
    % display sub images
    figure(8), imshow(sub_automated)
    figure(9), imshow(sub_automated)
    %Step 3: Do Normalized Cross-Correlation and Find Coordinates of Peak
    %Calculate the normalized cross-correlation and display it as a surface plot.
    % The peak of the cross-correlation matrix occurs where the sub_images are
    % best correlated. normxcorr2 only works on grayscale images, so we pass it
    % the red plane of each sub image.
    c = normxcorr2(sub_automated(:,:,1),sub_manual(:,:,1));
    figure (10), surf(c), shading flat
    %Step 4: Find the Total Offset Between the Images
    %The total offset or translation between images depends on the location
    %of the peak in the cross-correlation matrix, and on the size and position 
    %of the sub images.
    % offset found by correlation
    [max_c, imax] = max(abs(c(:)));
    [ypeak, xpeak] = ind2sub(size(c),imax(1));
    corr_offset = [(xpeak-size(sub_automated,2))
                   (ypeak-size(sub_automated,1))];
    % relative offset of position of subimages
    rect_offset = [(rect_manual(1)-rect_automated(1))
                   (rect_manual(2)-rect_automated(2))];
    % total offset
    offset = corr_offset + rect_offset;
    xoffset = offset(1);
    yoffset = offset(2);
    %Step 5: See if the Onion Image was Extracted from the Peppers Image
    %Figure out where onion falls inside of peppers.
    xbegin = round(xoffset+1);
    xend   = round(xoffset+ size(img_automated_gray,2));
    ybegin = round(yoffset+1);
    yend   = round(yoffset+size(img_automated_gray,1));
    % extract region from peppers and compare to onion
    extracted_automated =img_manual(ybegin:yend,xbegin:xend,:);
    if isequal(img_automated_gray,extracted_automated)
       disp('extracted_automated.png was extracted from img_automated.png')
    end
    %Step 6: Pad the Onion Image to the Size of the Peppers Image
    %Pad the automated image to overlay on manual, using the offset determined above.
    recovered_automated = uint8(zeros(size(img_manual)));
    recovered_onion(ybegin:yend,xbegin:xend,:) = img_automated_gray;
    figure(11), imshow(recovered_automated)
  figure (12), imshowpair(img_manual(:,:,1),recovered_automated,'blend')

