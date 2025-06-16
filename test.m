[noise1, Fs_n1] = audioread("babble_noise.wav");
[noise2, Fs_n2] = audioread("Speech_shaped_noise.wav");
[noise3, Fs_n3] = audioread("aritificial_nonstat_noise.wav");
[speech, Fs_s] = audioread("clean_speech.wav");
[speech2, Fs_s2] = audioread("clean_speech_2.wav");

%sound(noise1(1:1e5), Fs_n)



