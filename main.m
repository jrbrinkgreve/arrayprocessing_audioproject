filename = 'clean_speech.wav';
load("impulse_responses.mat")  %luckily Fs is the same for all samples!
[noise_babble, Fs] = audioread("babble_noise.wav");
[noise_speech_shaped, ~] = audioread("Speech_shaped_noise.wav");
[noise_art, ~] = audioread("aritificial_nonstat_noise.wav");
[speech_1, ~] = audioread("clean_speech.wav");
[speech_2, ~] = audioread("clean_speech_2.wav");




%choose whether to use clean sample 1 or 2:
speech_sample = speech_1;  %or speech_2


%normalize
speech_sample = speech_sample / max(abs(speech_sample));
noise_babble = noise_babble / max(abs(noise_babble));
noise_speech_shaped = noise_speech_shaped / max(abs(noise_speech_shaped));
noise_art = 0.05 * noise_art / max(abs(noise_art)); %as max == 0.01, this is really rough.



snr = 0;



inter1 = noise_babble(1:length(speech_sample));
inter2 = noise_speech_shaped(1:length(speech_sample));
inter3 = noise_art(1:length(speech_sample));


%get the data for all mics
num_mics = min(size(h_target));

%preallocate
X_target = zeros(num_mics, length(speech_sample));
X_inter1 = zeros(num_mics, length(speech_sample));
X_inter2 = zeros(num_mics, length(speech_sample));
X_inter3 = zeros(num_mics, length(speech_sample));



%do the convolutıons and store the sum in X signal
for i = 1:num_mics
    X_target(i,:) = conv(speech_sample, h_target(i,:), 'same') ;
    X_inter1(i,:) = conv(inter1, h_inter1(i,:), 'same') ;
    X_inter2(i,:) = conv(inter2, h_inter2(i,:), 'same') ;
    X_inter3(i,:) = conv(inter3, h_inter3(i,:), 'same') ;
end



X_noise = X_inter1 + X_inter2+ X_inter3;
X = X_target + 10^(-snr/20)*X_noise;




%Jasper paramters (I feel like longer time intervals would be ok)
tau = 20e-3; %20 ms windows for semi-stationarity
window_length = round(tau * Fs); %length of the Hanning window
hop_size = window_length / 2;  %50% overlap
nfft = window_length;       %FFT size



num_samples = length(X(1,:));  %assuming all entries are the same length
num_frames = floor((num_samples - window_length) / hop_size) + 1;
num_frames_no_speech = 1 / tau;  %1 = one second




%zero pad the function to ensure all the windows are completed
padded_length = (num_frames)*hop_size + window_length;
if num_samples < padded_length
    X = [X, zeros(num_mics, padded_length - num_samples)];
    X_noise = [X_noise, zeros(num_mics, padded_length - num_samples)];

end


% Windowing
w = hann(window_length, 'periodic')';  % Row vector
disp('Performing STFT...')

% STFT (Multichannel)
X_FFT = zeros(num_mics, nfft, num_frames);
for n = 1:num_mics
    for l = 1:num_frames
        idx = (1:window_length) + (l-1)*hop_size;
        frame = X(n, idx) .* w;               % Apply window
        X_FFT(n, :, l) = fft(frame, nfft);    % Store FFT
    end
end





%preallocate
atf_mat = zeros(4,320);
Rn_all = zeros(num_mics, num_mics, nfft);  % pre-allocate covariance matrices per freq
Rx_all = zeros(num_mics, num_mics, nfft);  % pre-allocate covariance matrices per freq
Rx_timevarying = zeros(num_mics, num_mics, nfft, num_frames);



disp('Estimating Rx and Rn...')   %calculate a [nummics x nummics x nfft] Rx tensor, averaged over all l
parfor j = 1:nfft
    Rx = zeros(num_mics, num_mics);
    for t = 1:num_frames
        x_kl = X_FFT(:, j, t);   
        Rx = Rx + (x_kl * x_kl');
    end
    Rx_all(:, :, j) = Rx / num_frames;
end




%To test out time varying rx?
%{
smoothing_factor = 0.9;  % Define a smoothing factor for the covariance estimation
Rx_timevarying(:,:,:,1) = Rx_all(:,:,:);  % Initialize the first frame of Rx_timevarying

for k = 1:nfft
    for l = 2:num_frames
        outer_product = X_FFT(:,k,l) * X_FFT(:,k,l)';
        Rx_timevarying(:,:,k,l) = (1 - smoothing_factor) * Rx_timevarying(:,:,k,l-1) + smoothing_factor * outer_product;
      
    end
end
%}


for j = 1:nfft
    Rn = zeros(num_mics, num_mics);
    for t = 1:num_frames_no_speech
        n_kl = X_FFT(:, j, t); %we know there is no speech in the first num_frames_no_speech
        Rn = Rn + (n_kl * n_kl');
    end
    Rn_all(:, :, j) = Rn / num_frames_no_speech;
end







% for j=1:nfft
%     K = Rx_all(:,:,j);
%     [U_k, sigma_k, V_k] = eig(K);
%     atf_mat(:, j) = U_k(:,1) / (U_k(1,1));
% end



%{

disp('Prewhitening...')
parfor j=1:nfft
    Rn_half = sqrtm(Rn_all(:,:,j));
    R_invhalf = pinv(Rn_half);
    Rx = Rx_all(:,:,j);

    K = R_invhalf * Rx * R_invhalf;

    [U_k, sigma_k, V_k] = eig(K);
    eigvals = diag(sigma_k);
    [~, sort_idx] = sort(real(eigvals), 'descend');
    V_k_sorted = V_k(:, sort_idx);
    
    atf_mat(:, j) = V_k_sorted(:,1) / V_k_sorted(1,1);  %RTF
end

%}

disp('GEVD...')
parfor j = 1:nfft
    [U_k, sigma_k, V_k] = eig(Rx_all(:,:,j), Rn_all(:,:,j));
    eigvals = diag(sigma_k);
    [~, sort_idx] = sort(real(eigvals), 'descend');
    V_k_sorted = V_k(:, sort_idx);
    
    atf_mat(:, j) = V_k_sorted(:,1) / V_k_sorted(1,1);  %RTF
end


%{
parfor j=1:nfft
    [U_k, sigma_k, V_k] = eig(Rn_all(:,:,j), Rx_all(:,:,j));
    atf_mat(:, j) = U_k(:,1) / (U_k(1,1));  %make it relative atf
end
%}


%MVDR Beamformer application
S_batches = zeros(nfft,num_frames);
disp('Beamforming...')

%precalc invs
Rn_pinvs = pagepinv(Rn_all);



%to test out timevarying Rx
%Rx_timevarying_pinvs = pagepinv(Rx_timevarying);



parfor i = 1: num_frames
    for j = 1:nfft 
        a_kl = atf_mat(:, j);
        x_kl =  X_FFT(:,j,i);

        w_mvdr = (Rn_pinvs(:,:,j) *a_kl ) / (a_kl' * Rn_pinvs(:,:,j) * a_kl);
        s_kl = w_mvdr' * x_kl;
        S_batches(j,i) = s_kl;


    end
end

%{



disp('Delay and Sum Beamforming...')

parfor i = 1:num_frames
    for j = 1:nfft 
        a_kl = atf_mat(:, j);            % steering vector
        x_kl = X_FFT(:, j, i);           % multichannel FFT input

        % Normalize steering vector (optional but often done)
        a_kl = a_kl / norm(a_kl);        

        % Delay and Sum Beamformer output
        s_kl = a_kl' * x_kl;             % weighted sum of aligned channels
        S_batches(j, i) = s_kl;
    end
end




%}



% iSTFT (Single channel — beamformed S_batches)
disp('Reconstructing...')
output_length = (num_frames - 1) * hop_size + window_length;
s_rec = zeros(1, output_length);
window_correction = zeros(1, output_length);
w = hann(window_length, 'periodic')';  % Reuse window
for l = 1:num_frames
    idx = (1:window_length) + (l-1)*hop_size;
    frame_fft = S_batches(:, l).';                    % Ensure row vector
    frame_time = real(ifft(frame_fft, nfft));         % Time-domain
    frame_time = frame_time(1:window_length) .* w;    % Apply window
    s_rec(idx) = s_rec(idx) + frame_time;             % Overlap-add
    window_correction(idx) = window_correction(idx) + w.^2;
end

% Normalize to correct window overlap energy
s_rec = s_rec ./ (window_correction + 1e-10);  % Avoid division by zero
s_rec(1:10) = 0;
s_rec(end-10:end) = 0;
s_rec = s_rec / max(abs(s_rec));


%sound(X(1,:),Fs)
%sound(s_rec(1:1e5),Fs)




% s_rec = X(1,1:length(s_rec));

% MSE is lowkey useless, we cant differentiate between good and bad
% reconstrruction with similar amplitudes
% mse = 0;
% for i = 1:length(s_rec)
%     mse = mse + (s_rec(i)-speech_sample(i))^2;
% 
% end
% 
% mse/length(s_rec)



s = speech_sample(1:length(s_rec));
s = s / max(abs(s));

% intrusive speech quality measure from: Speech Quality Assessment - Grancharov & kleijn
e = zeros(1,length(s_rec));
for i = 1:length(s_rec)
    e(i) = s(i)-s_rec(1,i);
end

snrerror = 10*log((s'*s)/(e*e'))




%stoi index:

length_difference = abs(length(s_rec(1,:)) - length(speech_1));

stoi_score = stoi(s_rec(1,:), speech_1(length_difference+1:end), Fs) %for best alignment




%

