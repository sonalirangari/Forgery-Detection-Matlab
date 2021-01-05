clc;
clear all;
close all;

%% Read Complete Audio File
% Create a WAVE (.wav) file in the current folder.
%Simple Audio Processing and Noise Mixing and Recovering Using Matlab
%Ask for an Audio File and Play
[fname, path] = uigetfile('*.*','Plaese Enter Message Audio');
fname=strcat(path,fname);
[w,Fs]=audioread(fname);
player = audioplayer(w,Fs)
play(player)
%sound(w,2*Fs);
%%
%Display Audio Signal
plot(w)
xlabel('Time')
ylabel('Audio Signal')
title('Original Signal');
%%
%Preprosessing 1.Normalization
[l ch] = size(w);
%%
%Encryption : Linear Encryption
key=32000
    for(i=1:ch)
        w1=w(:,i);
        mx=max(abs(w1));
        w1=w1/mx;
        w(:,i)=w1;
    end  
disp('Signal Normalization');
figure,
plot(w)

%%
%Encryption
En = floor(w*key);
plot(En)
[w,Fs]=audioread(fname);
player = audioplayer(En,Fs)
play(player)
%%
%Decryption
Dec=En/key;
player = audioplayer(Dec,Fs)
play(player)

plot(En(:,1),'black-')
hold on
plot(Dec(:,1),'red-')
legend('Encryption','Decryption');
xlabel('Time')
ylabel('Audio Signal')
title('Encryption and Decryption  of the signal')

%%
%%%%%%  fourier transform of vectorized signal fft() function  spectrum of an audio signal%%%%%%%
[xn fs]=audioread('e.wmv');
nf=1024; %number of point in DTFT
Y = fft(xn,nf);
f = fs/2*linspace(0,1,nf/2+1);
figure(3)
plot(f,abs(Y(1:nf/2+1)));
%%
% %Watermarking an image into an audio file using Least Significant Bit (LSB) method
% % extract watermark (the image) from the audio file
% input:   host_new(watermarked audio file)
% output:  wm(watermark image)
%%%% load data
wm_sz     = 20000;                        % watermark size
px_sz     = wm_sz/8;                      % number of pixels
im_sz     = sqrt(px_sz);                  % image size
host_new  = audioread ('e.wmv');   % new (watermarked) host signal
host_new  = uint8(255*(host_new + 0.5));  % double [-0.5 +0.5] to 'uint8' [0 255]
%% prepare host
host_bin  = dec2bin(host_new, 8);         % binary host [n 8]
%% extract watermark
wm_bin_str = host_bin(1:wm_sz, 8);
wm_bin    = reshape(wm_bin_str, px_sz , 8);
wm_str    = zeros(px_sz, 1, 'uint8');
for i     = 1:px_sz                        % extract water mark from the first plane of host               
wm_str(i, :) = bin2dec(wm_bin(i, :));      % Least Significant Bit (LSB)
end
wm= reshape(wm_str, im_sz , im_sz);
%% show image
figure(4),
imshow(wm)
title('WaterMark Audio')


%%
%%%%%%Sound Analysis with Matlab Implementation%%%%
%Time and frequency analysis, measurement of the crest factor, the dynamic range, etc.
[x, fs] = audioread('e.wmv');   % load an audio file
x = x(:, 1);                        % get the first channel
N = length(x);                      % signal length
t = (0:N-1)/fs;                     % time vector
% plot the signal waveform
figure(5)
plot(t, x, 'r')
xlim([0 max(t)])
ylim([-1.1*max(abs(x)) 1.1*max(abs(x))])
grid on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
xlabel('Time, s')
ylabel('Amplitude')
title('The signal in the time domain')
% plot the signal spectrogram
figure(6)
spectrogram(x, 1024, 3/4*1024, [], fs, 'yaxis')
box on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
xlabel('Time, s')
ylabel('Frequency, Hz')
title('Spectrogram of the signal')
h = colorbar;
set(h, 'FontName', 'Times New Roman', 'FontSize', 12)
ylabel(h, 'Magnitude, dB')
% spectral analysis
w = hanning(N, 'periodic');
[X, f] = periodogram(x, w, N, fs, 'power');
X = 20*log10(sqrt(X)*sqrt(2));
% plot the signal spectrum
figure(7)
semilogx(f, X, 'r')
xlim([0 max(f)])
grid on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
title('Amplitude spectrum of the signal')
xlabel('Frequency, Hz')
ylabel('Magnitude, dB')
% plot the signal histogram
figure(8)
histogram(x)
xlim([-1.1*max(abs(x)) 1.1*max(abs(x))])
grid on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
xlabel('Signal amplitude')
ylabel('Number of samples')
title('Probability distribution of the signal')
% autocorrelation function estimation
[Rx, lags] = xcorr(x, 'coeff');
d = lags/fs;
% plot the signal autocorrelation function
figure(9)
plot(d, Rx, 'r')
grid on
xlim([-max(d) max(d)])
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
xlabel('Delay, s')
ylabel('Autocorrelation coefficient')
title('Autocorrelation of the signal')
line([-max(abs(d)) max(abs(d))], [0.05 0.05],...
     'Color', 'k', 'LineWidth', 2, 'LineStyle', '--')
% compute and display the minimum and maximum values
maxval = max(x);
minval = min(x);
disp(['Max value = ' num2str(maxval)])
disp(['Min value = ' num2str(minval)])
 
% compute and display the the DC and RMS values
u = mean(x);
s = std(x);
disp(['Mean value = ' num2str(u)])
disp(['RMS value = ' num2str(s)])
% compute and display the dynamic range
D = 20*log10(maxval/min(abs(nonzeros(x))));
disp(['Dynamic range D = ' num2str(D) ' dB'])
% compute and display the crest factor
Q = 20*log10(maxval/s);
disp(['Crest factor Q = ' num2str(Q) ' dB'])
% compute and display the autocorrelation time
ind = find(Rx>0.05, 1, 'last');
RT = (ind-N)/fs;
disp(['Autocorrelation time = ' num2str(RT) ' s'])
%%
%%% The  Euclidean distance for the calculation is given below 
Im1 = audioread('d.mp4');
Im2 = audioread('A3.wav');

%the code for conversion of image to its normalized histogram
hn1 = imhist(Im1)./numel(Im1);
hn2 = imhist(Im2)./numel(Im2);

% Calculate the Euclidean distance
E_distance = sum(sqrt(hn1 - hn2).^2)

%%%% The  Manhattan distance for the calculation is given below
Im3 = audioread('d.mp4');
Im4 = audioread('A3.wav');

%the code for conversion of image to its normalized histogram
 hn1 = imhist(Im3)./numel(Im3);
hn2 = imhist(Im4)./numel(Im4);

% Calculate the Manhattan distance
M_distance = sum(abs(hn1 - hn2))
