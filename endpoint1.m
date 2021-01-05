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
% %VAD
% [clean, fs]=audioread('A2.wav');
% [masker, fs]=audioread('HaA1.wav');
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
% % % extract a few seconds from a given audio file
% [x,fs]= audioread('C:\Sonali Matlab\Matlab\Audio\HaA1.wav');
% 
% [m,n]=size(x);
% 
% dt=1/fs;
% 
% t=dt*(0:m-1);
% 
% figure(1);
% 
% plot(t,x);


%% load data
wm_sz     = 20000;                        % watermark size
px_sz     = wm_sz/8;                      % number of pixels
im_sz     = sqrt(px_sz);                  % image size
host_new  = audioread ('A2.wav');   % new (watermarked) host signal
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
figure(1),
imshow(wm)