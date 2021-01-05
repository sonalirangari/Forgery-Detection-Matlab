% noise = audioread('A2.wav');
% % noise           = audioread("WashingMachine-16-8-mono-200secs.wav");
% noise           = resample(noise,2,1);
% audioValidation=noise;
% audioValidation = audioValidation(1:numel(noise));
% 
% noise = 10^(-SNR/20) * noise * norm(audioValidation) / norm(noise);
% audioValidationNoisy = audioValidation + noise;
% audioValidationNoisy = audioValidationNoisy / max(abs(audioValidationNoisy));
%%
% load mtlb
% 
% a1 = mtlb(round(0.15*Fs):round(0.25*Fs));
% a2 = mtlb(round(0.37*Fs):round(0.45*Fs));
% 
% subplot(2,1,1)
% plot((0:numel(a1)-1)/Fs+0.15,a1)
% title('a_1')
% subplot(2,1,2)
% plot((0:numel(a2)-1)/Fs+0.37,a2)
% title('a_2')
% xlabel('Time (seconds)')

%%
% The  Euclidean distance for the calculation is given below 
Im1 = audioread('A2.wav');
Im2 = audioread('A3.wav');
 figure

%plotting of them
% plot(Im1)
% xlabel('Time')
% ylabel('Audio Signal')
% figure(1),
% title('Original Signal');
% 
% plot(Im2)
% xlabel('Time')
% ylabel('Audio Signal')
% figure(2),
% title('Forged Signal');

 plot(Im1)
subplot(2,2,1);
%  imshow(Im1);
xlabel('Time')
ylabel('Audio Signal')
title('Original Signal');

%  plot(Im2)
subplot(2,2,2);
% imshow(Im2);
xlabel('Time')
ylabel('Audio Signal')
title('Forged Signal');
% Im1=rgb2gray(Im1);
% Im2=rgb2gray(Im2);
%the code for conversion of image to its normalized histogram

hn1 = imhist(Im1)./numel(Im1);
hn2 = imhist(Im2)./numel(Im2);

% Calculate the Euclidean distance
E_distance = sum(sqrt(hn1 - hn2).^2)
%f=norm(hn1,hn2);


%%
% The  Manhattan distance for the calculation is given below

Im3 = audioread('A2.wav');
Im4 = audioread('A3.wav');
 figure
% figure(10),
%plotting of th1em
% subplot(1,2,1);
% imshow(Im1);
% subplot(1,2,2);
% imshow(Im2);
% Im1=rgb2gray(Im1);
% Im2=rgb2gray(Im2);
%the code for conversion of image to its normalized histogram

% plot(Im3)
subplot(2,2,1);
% imshow(Im1);
xlabel('Time')
ylabel('Audio Signal')
title('Original Signal');

% plot(Im4)
subplot(2,2,2);
% imshow(Im2);
xlabel('Time')
ylabel('Audio Signal')
title('Forged Signal')


 hn1 = imhist(Im3)./numel(Im3);
hn2 = imhist(Im4)./numel(Im4);

% Calculate the Manhattan distance
M_distance = sum(abs(hn1 - hn2))
%%

