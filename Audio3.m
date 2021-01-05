% clc
% clear 
% close all
% 
% % disp('Reading the file to be Tested');
% % 
% % [s fs]=wavread('C:\Sonali Matlab\Matlab\Audio\A1 1MG.wav');
% 
% %%
% % Read Complete Audio File
% % Create a WAVE file from the example file handel.mat, and read the file back into MATLAB®.
% % Create a WAVE (.wav) file in the current folder.
% 
%   load handel.mat
% 
% filename = 'A1.wav';
% audiowrite(filename,y,Fs);
% clear y Fs
% 
% info = audioinfo('A1.wav')
% 
% % Read the data back into MATLAB using audioread.
% [y,Fs] = audioread('A1.wav');
% [y1,Fs1] = audioread('A1.wav');
% 
% % Play the audio.
% sound(y,Fs);
% 
% % Plot Audio Data
% % Create a vector t the same length as y, that represents elapsed time.
% 
% plot(y)
% xlabel('Time')
% ylabel('Audio Signal')
%%
% % %VAD
% [clean, fs]=audioread('A2.wav');
% [masker, fs]=audioread('A2.wav');
% [masked, masker]=addnoise(clean,masker,12);
% plotWave_YW(1,masked, fs, 'freq',2,'signal',0);
% figure,
% [num, pos]=vad_YW(masked, fs,1,0.1,1,100);
%%
%Frequency-Domain Voice Activity Detection and Cepstral Feature Extraction%%%%%%%
% Create a dsp.AudioFileReader System object to read from an audio file.
% 
% fileReader = dsp.AudioFileReader('A2.wav');
% fs = fileReader.SampleRate;
% % Process the audio in 30 ms frames with a 10 ms hop. Create a default dsp.AsyncBuffer object to manage overlap between audio frames.
% 
% samplesPerFrame = ceil(0.03*fs);
% samplesPerHop = ceil(0.01*fs);
% samplesPerOverlap = samplesPerFrame - samplesPerHop;
% 
% fileReader.SamplesPerFrame = samplesPerHop;
% buffer = dsp.AsyncBuffer;
% % Create a voiceActivityDetector System object and a cepstralFeatureExtractor System object. Specify that they operate in the frequency domain. Create a dsp.SignalSink to log the extracted cepstral features.
% 
% VAD = voiceActivityDetector('InputDomain','Frequency');
% cepFeatures = cepstralFeatureExtractor('InputDomain','Frequency','SampleRate',fs,'LogEnergy','Replace');
% sink = dsp.SignalSink;
% % In an audio stream loop:
% % Read one hop's of samples from the audio file and save the samples into the buffer.
% % Read a frame from the buffer with specified overlap from the previous frame.
% % Call the voice activity detector to get the probability of speech for the frame under analysis.
% % If the frame under analysis has a probability of speech greater than 0.75, extract cepstral features and log the features using the signal sink. If the frame under analysis has a probability of speech less than 0.75, write a vector of NaNs to the sink.
% 
% threshold = 0.75;
% nanVector = nan(1,13);
% while ~isDone(fileReader)
%     audioIn = fileReader();
%     write(buffer,audioIn);
%     
%     overlappedAudio = read(buffer,samplesPerFrame,samplesPerOverlap);
%     X = fft(overlappedAudio,2048);
%     
%     probabilityOfSpeech = VAD(X);
%     if probabilityOfSpeech > threshold
%         xFeatures = cepFeatures(X);
%         sink(xFeatures')
%     else
%         sink(nanVector)
%     end
% end
% % Visualize the cepstral coefficients over time.
% 
% timeVector = linspace(0,15,size(sink.Buffer,1));
% plot(timeVector,sink.Buffer)
% xlabel('Time (s)')
% ylabel('MFCC Amplitude')
% legend('Log-Energy','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12')
%%
% %Detect Voice Activity 
% fileReader = dsp.AudioFileReader('HaA1.wav');
% fs = fileReader.SampleRate;
% fileReader.SamplesPerFrame = ceil(10e-3*fs);
% % Create a default voiceActivityDetector System object to detect the presence of speech in the audio file.
% 
% VAD = voiceActivityDetector;
% voiceActivityDetector=fileReader.SamplesPerFrame;
% % Create a scope to plot the audio signal and corresponding probability of speech presence as detected by the voice activity detector. Create an audio device writer to play the audio through your sound card.
% 
% scope = dsp.TimeScope( ...
%     'NumInputPorts',2, ...
%     'SampleRate',fs, ...
%     'TimeSpan',3, ...
%     'BufferLength',3*fs, ...
%     'YLimits',[-1.5 1.5], ...
%     'TimeSpanOverrunAction','Scroll', ...
%     'ShowLegend',true, ...
%     'ChannelNames',{'Audio','Probability of speech presence'});
% deviceWriter = audioDeviceWriter('SampleRate',fs);
% % In an audio stream loop:
% % Read from the audio file.
% % Calculate the probability of speech presence.
% % Visualize the audio signal and speech presence probability.
% % Play the audio signal through your sound card.
% 
% while ~isDone(fileReader)
%     audioIn = fileReader();
%     probability = VAD(audioIn);
%     scope(audioIn,probability*ones(fileReader.SamplesPerFrame,1))
%     deviceWriter(audioIn);
% end

%%
%Detect Voice Activity Using Overlapped Frames
% afr = audioread('A2.wav');
% fs = afr.SampleRate;
% 
% frameSize = ceil(20e-3*fs);
% overlapSize = ceil(0.75*frameSize);
% hopSize = frameSize - overlapSize;
% afr.SamplesPerFrame = hopSize;
% % dsp.AsyncBuffer=afr.SamplesPerFrame;
% 
% inputBuffer = dsp.AsyncBuffer('Capacity',frameSize);
% 
% VAD = voiceActivityDetector('FFTLength',1024);
% 
% scope = dsp.TimeScope('NumInputPorts',2, ...
%     'SampleRate',fs, ...
%     'TimeSpan',3, ...
%     'BufferLength',3*fs, ...
%     'YLimits',[-1.5,1.5], ...
%     'TimeSpanOverrunAction','Scroll', ...
%     'ShowLegend',true, ...
%     'ChannelNames',{'Audio','Probability of speech presence'});
% 
% player = audioDeviceWriter('SampleRate',fs);
% 
% pHold = ones(hopSize,1);
% 
% while ~isDone(afr)
%     x = afr();
%     n = write(inputBuffer,x);
% 
%     overlappedInput = read(inputBuffer,frameSize,overlapSize);
% 
%     p = VAD(overlappedInput);
% 
%     pHold(end) = p;
%     scope(x,pHold)
% 
%     player(x);
% 
%     pHold(:) = p;
% end
% 
% release(player)
%%
% Read in an audio file and convert it to a frequency representation.

% [audioIn,fs] = audioread('A2.wav');
% 
% win = hann(1024,'periodic');
% S = stft(audioIn,'Window',win,'OverlapLength',512,'Centered',false);
% % To extract the mel-frequency cepstral coefficients, call mfcc with the frequency-domain audio. Ignore the log-energy.
% 
% coeffs = mfcc(S,fs,'LogEnergy','Ignore');
% 
% %In many applications, MFCC observations are converted to summary statistics for use in classification tasks. 
% nbins = 60;
% for i = 1:size(coeffs,2)
%     figure
%     histogram(coeffs(:,i),nbins,'Normalization','pdf')
%     title(sprintf('Coefficient %d',i-1))
% end
%%
% audioSource = dsp.AudioFileReader('SamplesPerFrame',80,...
%                                'Filename','A2.wav',...
%                                'OutputDataType', 'single');
% % Note: You can use a microphone as a source instead by using an audio
% % device reader (NOTE: audioDeviceReader requires an Audio Toolbox
% % (TM) license)
% % audioSource = audioDeviceReader('OutputDataType', 'single', ...
% %                              'NumChannels', 1, ...
% %                              'SamplesPerFrame', 80, ...
% %                              'SampleRate', 8000);
% % Create a time scope to visualize the VAD decision (channel 1) and the
% % speech data (channel 2)
% scope = dsp.TimeScope(2, 'SampleRate', [8000/80 8000], ...
%                       'BufferLength', 80000, ...
%                       'YLimits', [-0.3 1.1], ...
%                       'ShowGrid', true, ...
%                       'Title','Decision speech and speech data', ...
%                       'TimeSpanOverrunAction','Scroll');
%                   
% % Initialize VAD parameters
% VAD_cst_param = vadInitCstParams;
% clear vadG729
% % Run for 10 seconds
% numTSteps = 1000;
% while(numTSteps)
%   % Retrieve 10 ms of speech data from the audio recorder
%   speech = audioSource();
%   % Call the VAD algorithm
%   decision = vadG729(speech, VAD_cst_param);
%   % Plot speech frame and decision: 1 for speech, 0 for silence
%   scope(decision, speech);
%   numTSteps = numTSteps - 1;
% end
% release(scope);                  


%%
% t = [1:0.01:5];                                                         % Time Vector
% y = -2+2.*sin(2*pi*t);                                                        % Signal
% zci = @(v) find(v(:).*circshift(v(:), [-1 0]) <= 0);                    % Returns Zero-Crossing Indices Of Argument Vector
% zx = zci(y+2);                                                            % Approximate Zero-Crossing Indices
% figure(1)
% plot(t, y, '-r')
% hold on
% plot(t(zx), y(zx), 'bp')
% hold off
% grid
% legend('Signal', 'Approximate Zero-Crossings')

%%

%%
% clc;
% clear;
% close all;
% [Audio,fs] = audioread('A2.wav');
% %%
% realS =[13575 29531 42639 58180 73148 86311 100846  115201 131129  146475];
% realF=[150366 135855 120426 105256  93027  79042  63947  49655 34330 19901];
% %%
% nA = numel(Audio);
% Audio=awgn(Audio,50);
% t = 1:nA;
% eA = zeros(nA,1);
% N10ms=fs*10e-3;  % Number of Samples in 10 ms
%  for i = N10ms+1:nA-N10ms
%      eA(i) = sum(abs(Audio(i-N10ms:i+N10ms)));
%  end
%  for i=1:N10ms
%      eA(i)=sum(abs(Audio(1:i+N10ms)));
%  end
%  for i=nA-N10ms:nA
%      eA(i)=sum(abs(Audio(i-N10ms:nA)));
%  end
% %%
% zcr=i;
% for i=1:160:10000
%     IZCb(i)=zcr(Audio(i:i+159));
% end
% IZC_bar=mean(IZCb)
% IF=25/N10ms; % 25 crosses in 10ms
% IZCT=min(IF, IZC_bar+2*std(IZCb))
% IMX=max(eA);
% IMN=min(eA);
% l1=0.03*(IMX-IMN)+IMN;
% l2=4*IMN;
% ITL=min(l1,l2)
% ITU=5*ITL
% %% compute starting point
% S=[];
% frameSize=8000;
% for i=1:frameSize:nA-frameSize
%     start=finder(Audio(i:i+frameSize),eA(i:i+frameSize),ITU,ITL,IZCT);
%     if ~isempty(start)
%   start=start+i;
%     end
%     S=[S,start];
% end
% %% compute ending point
% Audio=flipud(Audio);
% F=[];
% eA=flipud(eA);
% for i=1:frameSize:nA-frameSize
%     start=finder(Audio(i:i+frameSize),eA(i:i+frameSize),ITU,ITL,IZCT);
%     if ~isempty(start)
%   start=start+i;
%     end
%     F=[F,nA-start];
% end   
% plot(flipud(Audio))
% hold on
% plot(S,zeros(size(S)),'k^','MarkerFaceColor','k');
% plot(F,zeros(size(F)),'ro','MarkerFaceColor','r');
% legend('Signal','Start point','End point')
% %%
% for i=1:numel(S)
%     value(i)=min(dist(S(i),realS));
% end
% AVGstartError=mean(value)
% for i=1:numel(F)
%     value(i)=min(dist(F(i),realF));
% end
% AVGEndError=mean(value)

%%
% close all;
% %{Step 0: Reading the File & initializing the Time and Freq.
%     [x,fs]=audioread('HaA1.wav');
%     x=x(:,1);
%     ts=1/fs;
%     N=length(x);
%     Tmax=(N-1)*ts;
%     fsu=fs/(N-1);
%     t=(0:ts:Tmax);
%     f=(-fs/2:fsu:fs/2);
%     figure, subplot(411),plot(t,x),xlabel('Time'),title('Original Speech');
%     subplot(412),plot(f,fftshift(abs(fft(x)))),xlabel('Freq (Hz)'),title('Frequency Spectrum');
% 
% %%Step 1: Pre-Emphasis
% 
%     a=1;
%     b=[1 -0.95];
%     y=filter(b,a,x);
%     subplot(413),plot(t,y),xlabel('Time'),title('Signal After High Pass Filter - Time Domain');
%     subplot(414),plot(f,fftshift(abs(fft(y)))),xlabel('Freq (Hz)'),title('Signal After High Pass Filter - Frequency Spectrum');
% 
% %%Step 2: Frame Blocking
% 
%     frameSize=256;
%     frameOverlap=128;
%     enframe=y;
%     frames=enframe(y,frameSize,frameOverlap);
%     NumFrames=size(frames,1);
% 
% %%Step 3: Hamming Windowing
% 
%     hamm=hamming(256)';
%     for i=1:NumFrames
%     windowed(i,:)=frames(i,:).*hamm;
%     end
% 
% %%Step 4: FFT 
% %Taking only the positive values in the FFT that is the first half of the frame after being computed. 
% 
%     for i=1:NumFrames
%     fft(i,:)=abs(windowed(i,:))*(frameSize/2);     
%     end
% 
% %%Step 5: Mel Filterbanks
% %Lower Frequency = 300Hz
% %Upper Frequency = fs/2
% %With a total of 22 points we can create 20 filters.
%     Nofilters=20;
%     lowhigh=[300 fs/2];
%     %Here logarithm is of base 'e'
%     lh_mel=1125*(log(1+lowhigh/700));
%     mel=linspace(lh_mel(1),lh_mel(2),Nofilters+2);
%     melinhz=700*(exp(mel/1125)-1);
%     %Converting to frequency resolution
%     fres=floor(((frameSize)+1)*melinhz/fs); 
%     %Creating the filters
%     for m =2:length(mel)-1
%         for k=1:frameSize/2
%      if k<fres(m-1)
%         H(m-1,k) = 0;
%     elseif (k>=fres(m-1)&&k<=fres(m))
%         H(m-1,k)= (k-fres(m-1))/(fres(m)-fres(m-1));
%     elseif (k>=fres(m)&&k<=fres(m+1))
%        H(m-1,k)= (fres(m+1)-k)/(fres(m+1)-fres(m));
%     elseif k>fres(m+1)
%         H(m-1,k) = 0;    
%      end 
%         end
%     end
%     %H contains the 20 filterbanks, we now apply it to the
%     %processed signal.
%    bankans=cell(NumFrames,Nofilters);
% for i=20:1
% for j=1:256
%         bankans{i,j}=sum(fft(:,i).* H(j,:));
%     end
%     end
% 
% 
% %%Step 6: Neutral Log and DCT
%     %pkg load signal
%     %Here logarithm is of base '10'
%     logged=log(bankans);
%     for i=1:NumFrames
% 
%         mfcc(i,:)=dct2(logged(i,:));
%     end
% 
% %plotting the MFCC
% 
%     figure 
%     hold on
%     for i=1:NumFrames
%         plot(lnd(i,:));
%     end
%     hold off
%%
%%%%%%Sound Analysis with Matlab Implementation%%%%
% clear, clc, close all
% % get a section of the sound file
% [x, fs] = audioread('A2.wav');   % load an audio file
% x = x(:, 1);                        % get the first channel
% N = length(x);                      % signal length
% t = (0:N-1)/fs;                     % time vector
% % plot the signal waveform
% figure(1)
% plot(t, x, 'r')
% xlim([0 max(t)])
% ylim([-1.1*max(abs(x)) 1.1*max(abs(x))])
% grid on
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% xlabel('Time, s')
% ylabel('Amplitude')
% title('The signal in the time domain')
% % plot the signal spectrogram
% figure(2)
% spectrogram(x, 1024, 3/4*1024, [], fs, 'yaxis')
% box on
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% xlabel('Time, s')
% ylabel('Frequency, Hz')
% title('Spectrogram of the signal')
% h = colorbar;
% set(h, 'FontName', 'Times New Roman', 'FontSize', 14)
% ylabel(h, 'Magnitude, dB')
% % spectral analysis
% w = hanning(N, 'periodic');
% [X, f] = periodogram(x, w, N, fs, 'power');
% X = 20*log10(sqrt(X)*sqrt(2));
% % plot the signal spectrum
% figure(3)
% semilogx(f, X, 'r')
% xlim([0 max(f)])
% grid on
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% title('Amplitude spectrum of the signal')
% xlabel('Frequency, Hz')
% ylabel('Magnitude, dB')
% % plot the signal histogram
% figure(4)
% histogram(x)
% xlim([-1.1*max(abs(x)) 1.1*max(abs(x))])
% grid on
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% xlabel('Signal amplitude')
% ylabel('Number of samples')
% title('Probability distribution of the signal')
% % autocorrelation function estimation
% [Rx, lags] = xcorr(x, 'coeff');
% d = lags/fs;
% % plot the signal autocorrelation function
% figure(5)
% plot(d, Rx, 'r')
% grid on
% xlim([-max(d) max(d)])
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% xlabel('Delay, s')
% ylabel('Autocorrelation coefficient')
% title('Autocorrelation of the signal')
% line([-max(abs(d)) max(abs(d))], [0.05 0.05],...
%      'Color', 'k', 'LineWidth', 2, 'LineStyle', '--')
% % compute and display the minimum and maximum values
% maxval = max(x);
% minval = min(x);
% disp(['Max value = ' num2str(maxval)])
% disp(['Min value = ' num2str(minval)])
%  
% % compute and display the the DC and RMS values
% u = mean(x);
% s = std(x);
% disp(['Mean value = ' num2str(u)])
% disp(['RMS value = ' num2str(s)])
% % compute and display the dynamic range
% D = 20*log10(maxval/min(abs(nonzeros(x))));
% disp(['Dynamic range D = ' num2str(D) ' dB'])
% % compute and display the crest factor
% Q = 20*log10(maxval/s);
% disp(['Crest factor Q = ' num2str(Q) ' dB'])
% % compute and display the autocorrelation time
% ind = find(Rx>0.05, 1, 'last');
% RT = (ind-N)/fs;
% disp(['Autocorrelation time = ' num2str(RT) ' s'])
% commandwindow
 %%
%   [wave1,fs1]=audioread('A2.wav'); % read file into memory */
% [wave2,fs2]=audioread('A3.wav');
% %sound(wave,fs); % see what it sounds like */
% t=0:1/fs1:(length(wave1)-1)/fs1; % and get sampling frequency */
% % figure(90);
% %           subplot(2,1,1)
% %           %plot(t,wave)
% %           plot(t,abs(wave1))
% %           title('Wave File')
% %           ylabel('Amplitude')
% %           xlabel('Length (in seconds)')
% 
% 
% L = length(wave1);
% NFFT = 2^nextpow2(L); % Next power of 2 from length of y
% Y = fft(wave1,NFFT)/L;
% Y2=fft(wave2,NFFT)/L;
% f = fs1/2*linspace(0,1,NFFT/2+1);
% 
% 
% % Plot single-sided amplitude spectrum.
% %         subplot(2,1,2)
%         plot(f,2*abs(Y(1:NFFT/2+1))) 
%         hold on
%         plot(f,2*abs(Y2(1:NFFT/2+1))) 
%         title('Single-Sided Amplitude Spectrum of y(t)')
%         xlabel('Frequency (Hz)')
%         ylabel('|Y(f)|')
%         legend('background','background+noise');
%  A = max(Y)
%%
% Frequency sweep
% F0 = 10;        % start frequency, Hertz
% F1 = 100;       % stop frequency, Hertz
% T  = 0.5;       % duration, seconds
% FS = 1000;      % sample rate, Hertz
% N = round(T * FS);
% t = T * (0:N-1)' / (N-1);
% y = sin(2 * pi * (F0 + (F1 - F0) / 2 .* t / T) .* t);
% subplot(2,1,1); plot(t, y); xlabel('seconds');
% %
% % Positive-slope zero-crossing detector
% z = and((y > 0), not(circshift((y > 0), 1)));  z(1) = 0;
% subplot(2,1,2); plot(t, z); xlabel('seconds');
% %
% % Find the locations of the zero-crossing points
% crossing_points = find(z);
%%
%%%%%%spectrum of an audio signal%%%%%%%
% % Read it in with wavread()
% [signal,fs] = audioread('HaA1.wav');
% % If signal is Nx2 (two columns), extract one of them
% signal = signal(:,1);
% % If you have the Signal Processing Toolbox, enter
% plot(psd(spectrum.periodogram,signal,'Fs',fs,'NFFT',length(signal)));

%%%%%%  fourier transform of vectorized signal fft() function  spectrum of an audio signal%%%%%%%
% [xn fs]=audioread('A2.wav');
% nf=1024; %number of point in DTFT
% Y = fft(xn,nf);
% f = fs/2*linspace(0,1,nf/2+1);
% figure(4),
% plot(f,abs(Y(1:nf/2+1)));
%%
% clc
% clear all;
% close all;
% [y,Fs]=audioread('A2.wav');
% YY=y;
% %  AUDIO_PLY_PLOT(y,Fs)
% plot(y);
% xlabel('Magnitude');
% ylabel('Frequency');
% title('Original Audio Signal');
% %% FILTER PROCESS
% FILT_SIGNAL = filter([1 -1 1], 1, y);
% %% EXTRACT THE CHANNELS FROM AUDIO SIGNALS
% y=FILT_SIGNAL;
% CHANNEL_1=(y(:,1));
% % CHANNEL_2=(y(:,2));
% %% SEGMENT THE AUDIO DATA
% % EXTRACT SAMPLES VIA EMD
% Ts=4000;
%  ADI_COS=y(1:10000);
% SEG=ADI_COS;
% figure,plot(SEG,'c');
% title('Segment Audio Signal')
% HIL_SPECTR_ANALY_EMD=SEG;
%  INT_MOD_FUN = HIL_SPECTR_ANALY_EMD(ADI_COS)
% %% ADD WATER MARK DATA 
% CELL_LENGTH=length(INT_MOD_FUN);
% % FIND FREQUENCY LEVEL OF EMD DATA'S
%  INC_MOD_FUN= HIL_SPECTR_ANALY_EMD(ADI_COS)
% %% TO FIND EXTREMA AND MINIMA POSITION 
% NN=INC_MOD_FUN{5};
% q = quantizer([4,3]);
% y1 = num2bin(q,SEG)
% %% FILTERING PROCESS 
% % 
% DECI_FACT=1;
% j=real(abs(INC_MOD_FUN{1}));
% y=AUDIO_FILTER(DECI_FACT,j);
% soundsc(y,Fs)
% %% TO FIND EXTREMA ELEMENT POSITION 
% %%
% %% RAM SIGNAL
% N = size(y,1);
% df = Fs / N;
% w = (-(N/2):(N/2)-1)*df;
% y = fft(y(:,1), N) / N; %//For normalizing, but not needed for our analysis
% y2 = fftshift(y);
% y3=ifftshift(y2);
% % soundsc(y1,Fs)
% %% 
% %% DISCREATE TIME ANALYTIC USING HILBERT TRANSFORM
% % 17 CELLS ARE ANALYTIC BELLOW
% for k = 1:length(INT_MOD_FUN)
%    AUDIO_DATA(k) = sum(INT_MOD_FUN{k}.*INT_MOD_FUN{k});
%    PHASE_ANGLE   = hilbert(INT_MOD_FUN{k});
%  
% end
% for k = 1:length(INT_MOD_FUN)
%  
%    PHASE_ANGLE   = angle(INT_MOD_FUN{k});
%    DIFF{k} = diff(PHASE_ANGLE)/Ts/(2*pi);
% end
% [u,v] = sort(-AUDIO_DATA);
% AUDIO_DATA     = 1-AUDIO_DATA/max(AUDIO_DATA);
% %% IMF FUNCTION PLOTTING PERFORMANCE
% figure,plot(INT_MOD_FUN{1})
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions1')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    1');
% %% 
% figure,plot(INT_MOD_FUN{1},'g')
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions2')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    2');
% %% 
% figure,plot(INT_MOD_FUN{3},'m')
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions 3')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    3');
% %% 
% figure,plot(INT_MOD_FUN{4},'k')
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions 4')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    1');
% %% 
% figure,plot(INT_MOD_FUN{5},'c')
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions1')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    5');
% %% 
% figure,plot(INT_MOD_FUN{6},'m')
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions 6')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    6');
% %% 
% figure,plot(INT_MOD_FUN{7},'c')
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions  7')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    7');
% %% 
% figure,plot(INT_MOD_FUN{8},'k')
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions  8')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    8');
% %% 
% figure,plot(INT_MOD_FUN{9},'b')
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions 9 ')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    9');
% %% 
% figure,plot(INT_MOD_FUN{10},'m')
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions 10')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    10');
% %% 
% figure,plot(INT_MOD_FUN{11},'r')
% xlabel('Tim')
% ylabel('Amplitude')
% title('Intrinsic Mode Functions 11')
% grid on
% h = gcf; 
% set(h,'Name','Intrinsic Mode Functions    11');
% %% 
% %% P=INT_MOD_FUN{1}
% PCM=[1 0 1 0 ];
% q = quantizer([4,3]);
% DATA=INT_MOD_FUN{1};
% L=length(DATA)
% CONS=num2bin(q,DATA)
% CONS(L+1:L+4)
% DA=bin2num(q,CONS)
% ORIGINAL=YY;
% EMBED_AUDIO=YY;
% EMBED_AUDIO(1:10000)=DA
% %% TO PLOT ORIGINAL AUDI SIGNALS  AND AUDIO WATERMARK DATA
% Y=EMBED_AUDIO;
% X=ORIGINAL;
% % FEATURES CALCULATION 
% %% PSNR RANGE FOR ORIGINAL AND WATERMARK DATA
% % Y=Y(1:512);
% % X=X(1:512);
% % filtered_signal=BAND_PASS_FILTER(input)
% % PSNR_RANG= PSNR(X,Y)
% SNR = SNR_RANGE(Y, X)
% figure,plot(EMBED_AUDIO,'g');
% title('WATER MARKED DATA');
% figure,plot(ORIGINAL,'k');
% title('ORIGINAL AUDIO DATA');
% %% WATERMARK DATA EXTRACTION PROCESS
% ADI_COS=EMBED_AUDIO(1:10000);   
% INC_MOD_FUN= HIL_SPECTR_ANALY_EMD(ADI_COS)
%  LN=length(INC_MOD_FUN);
%  
%  for i=1:LN
% RECONS_WATER_MARKED{i}=INC_MOD_FUN{i}
% RCON_WTR_MRKD=cell2mat(RECONS_WATER_MARKED)
%  end
%  %% COMPARE WITH ORIGINAL DATA
%  % FIND WATER MARK POSITION 
%  ORGNL=X;
%  RCON_WTR_MRKD(1:length(ADI_COS))=[]
%  X(1:length( RCON_WTR_MRKD))=[]  ;
%  % TO PLAY RECONSTRUCTED DATA
%  soundsc(X,Fs)
%     
%     
%     
%     %%  PSD is the distribution of power per unit frequency
%     % vector of normalized frequencies 
%  AD=X;
% SMPL_AD=AD(1:512);
% RND_GEN= randn(size( SMPL_AD));
% SMP= cos(2*pi* SMPL_AD*200);
% ADI_COS=SMP+  RND_GEN;
% PSD_DATA=spectrum.periodogram;
% figure,psd(PSD_DATA,ADI_COS,'Fs',Fs)
%     
%     %% 
%     EMD=INT_MOD_FUN{11};
%   
% SMPL_AD=   EMD(1:512);
% RND_GEN= randn(size( SMPL_AD));
% SMP= cos(2*pi* SMPL_AD*200);
% ADI_COS=SMP+  RND_GEN;
% PSD_DATA=spectrum.periodogram;
% figure,psd(PSD_DATA,ADI_COS,'Fs',Fs)
%%
% %Watermarking an image into an audio file using Least Significant Bit (LSB) method
% % extract watermark (the image) from the audio file
% input:   host_new(watermarked audio file)
% output:  wm(watermark image)
 clc
clear
close all 
%% load data
% wm_sz     = 20000;                        % watermark size
% px_sz     = wm_sz/8;                      % number of pixels
% im_sz     = sqrt(px_sz);                  % image size
% host_new  = audioread ('A2.wav');   % new (watermarked) host signal
% host_new  = uint8(255*(host_new + 0.5));  % double [-0.5 +0.5] to 'uint8' [0 255]
% %% prepare host
% host_bin  = dec2bin(host_new, 8);         % binary host [n 8]
% %% extract watermark
% wm_bin_str = host_bin(1:wm_sz, 8);
% wm_bin    = reshape(wm_bin_str, px_sz , 8);
% wm_str    = zeros(px_sz, 1, 'uint8');
% for i     = 1:px_sz                        % extract water mark from the first plane of host               
% wm_str(i, :) = bin2dec(wm_bin(i, :));      % Least Significant Bit (LSB)
% end
% wm= reshape(wm_str, im_sz , im_sz);
% %% show image
% figure(1),
% imshow(wm)
 %
%  insert watermark (the image) from the audio file
% clc
% clear
% close all 
%% load data
% [host, f] = audioread ('A2.wav');   % host signal
% host      = uint8(255*(host + 0.5));  % double [-0.5 +0.5] to 'uint8' [0 255]
% wm        = imread('1.png');  % watermark
% [r, c]    = size(wm);                 % watermark size
% wm_l      = length(wm(:))*8;          % watermark length
% %% watermarking
% if length(host) < (length(wm(:))*8)
%     disp('your image pixel is not enough')
% else
% %% prepare host
% host_bin  = dec2bin(host, 8);         % binary host [n 8]
% %% prepare watermark   
% wm_bin    = dec2bin(wm(:), 8);        % binary watermark [n 8]
% wm_str    = zeros(wm_l, 1);           % 1-D watermark [(n*8) 1]
% for j = 1:8                           % convert [n 8] watermark to [(n*8) 1] watermark
% for i = 1:length(wm(:))
% ind   = (j-1)*length(wm(:)) + i;
% wm_str(ind, 1) = str2double(wm_bin(i, j));
% end
% end
% %% insert watermark into the host
% for i     = 1:wm_l                   % insert water mark into the first plane of host               
% host_bin(i, 8) = dec2bin(wm_str(i)); % Least Significant Bit (LSB)
% end 
% %% watermarked host
% host_new  = bin2dec(host_bin);       % watermarked host
% host_new  = (double(host_new)/255 - 0.5);   % 'uint8' [0 255] to double [-0.5 +0.5]
% %% save the watermarked host
% audiowrite('A2.wav', host_new, f)     % save watermarked host ausio
% end
% %%
%%Watermark%%%%%%%%%%%%%%%%%%%
% close all;
% clc;
% %   read image 
% [embededimage_fname, image_pthname] = ...
%     uigetfile('*.jpg; *.png; *.tif; *.bmp', 'Select the Cover Image');
% if (embededimage_fname ~= 0)
%     embedded_image = strcat(image_pthname, embededimage_fname);
%     embedded_image = double( rgb2gray( imread( embedded_image ) ) );
%     embedded_image = imresize(embedded_image, [512 512], 'bilinear');
% else
%     return;
% end
% % read audio
% [watermarkAudio_fname, watermark_pthname] = ...
%     uigetfile('*.wav', 'Select the Watermark audio');
% if (watermarkAudio_fname ~= 0)
%     watermark_audio = strcat(watermark_pthname, watermarkAudio_fname);
%     [n, fs] = audioread(watermark_audio);            % "n" is number of  samples and fs is the sample rate 
% %    watermark_audio = imresize(watermark_audio, [512 512], 'bilinear');
% else
%     return;
% end
% imbin_seq = reshape(dec2bin(embedded_image, 8) - '0', 1, []);
% % To calculate modified mean
% na = 2^15;
% n = n*na;
% nLength = length(n);
% modified_mean = mean(abs(n));
% % Embedding Range
% lmda = 2.4;    %pick any value of lmda from the range (2.0, 2.5);
% e_range = ceil(lmda * modified_mean);
% % number of bins and Bin Width
% h = histogram(n);
% Bin_Num = h.NumBins();
% Bin_Width = h.BinWidth();
% 
