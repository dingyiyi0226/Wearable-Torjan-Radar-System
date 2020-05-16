clear; clc; close all;

%% Parameters
distance = [100 200];                % Object distance from Tx           (* 1e-2)
speed = [5000 10000 15000 20000 25000 30000 100000 200000 500000 1000000]; % Object rotation line velocity     (* 1e3)
radius = 0.14;                                  % Object radius
freq = 5.8e6;                                   % Tx center frequency               (* 1e-3)
fs_high = 2e7;                                  % [High] Sampling frequency
T_high = 1 / fs_high;                           % [High] Sampling time interval
fs_low = 2e5;                                   % [Low] Sampling frequency
T_low = 1 / fs_low;                             % [Low] Sampling time interval
ds_ratio = fs_high / fs_low;                    % Downsample ratio
t_start = 0;                                    % Simulation start time
t_end = 1;                                   % Simulation end time
N_high = fs_high * (t_end - t_start)+1;           % [High] Number of points
N_highTrunc = N_high;                           % Number of points after rps truncation
N_low = fs_low * (t_end - t_start)+1;             % [Low] Number of points
t_high = 0:1/fs_high:t_end;                     % [High] Time axis
f_high = 0:fs_high/N_high:fs_high;              % [High] Frequency axis
f_highTrunc = 0:fs_high/N_highTrunc:fs_high;    % [High] [RPS Truncated] Frequency axis
t_low = 0:1/fs_low:t_end;                       % [Low] Time axis
f_low = 0:fs_low/N_low:fs_low;                  % [Low] Frequency axis
f_lowTrunc = 0:fs_low/N_low:fs_low;        % [Low] [RPS Truncated] Frequency axis
bw = 1e6;                                       % Bandwidth                         (* 1e-2)
TimeResolu = 1000 * (1/fs_high);                % Constant for graph plotting
duty1 = 0.3;                                     % Duty cycle for speed generator
duty2 = 0.5;
duty3 = 0.7;
sweep_freq = 71.4;                               % Freq. Modulation (Up-Down interval 
                                                % for cont. triangular chirp)
tm = 1 / sweep_freq;                            % Freq. modulation (Up-Down interval 
                                                % for cont. triangular chirp)
sweep_num = floor(t_end * sweep_freq);                 % Constant for graph plotting

%% Options
bool_rpsModulation = 1;
bool_rpsTruncation = 1;
%% Simulation

% Tx Signal
[Tx_signal, down_part_chirp_tx] = chirper(freq, fs_high, t_high, sweep_freq, floor(sweep_num), bw);

% Rx Signal with (distance, speed) combinations
for i=1:numel(distance)
    pure_distance_Rx_signal = distanceShift(Tx_signal, distance(i), down_part_chirp_tx, fs_high);
    for j=1:numel(speed)
        %% Doppler Shift
        freq_dev = 2 * freq * speed(j) / (3e8);
        [Rx_signal, down_part_chirp_rx] = chirper(freq + freq_dev, fs_high, t_high, sweep_freq, floor(sweep_num), bw);
        Rx_copy_signal = Rx_signal;
        
        if bool_rpsModulation == 1
            rps = speed(j) / 1000 / (2 * pi * radius);
            Rx_duty1 = rpsModulation(Rx_signal, pure_distance_Rx_signal, t_high, duty1, rps);
            Rx_duty2 = rpsModulation(Rx_signal, pure_distance_Rx_signal, t_high, duty2, rps);
            Rx_duty3 = rpsModulation(Rx_signal, pure_distance_Rx_signal, t_high, duty3, rps);
            % figure
            % pspectrum(...
            %     Rx_signal, fs_high, 'spectrogram', ...
            %     'TimeResolution', TimeResolu, 'OverlapPercent', 10, 'Leakage', 0.85)
            % title('post rps Rx signal')
            % savefig(['PostRPSRxSignal_', num2str(distance(i)), '_', num2str(speed(j)), '.fig'])
            % close(gcf)
        end
        if bool_rpsTruncation == 1
            N_highTrunc = floor(floor(t_end*rps)*(fs_high/rps))+1;
            f_highTrunc = 0:fs_high/N_highTrunc:fs_high;
            Rx_duty1 = Rx_duty1(1:N_highTrunc);
            Rx_duty2 = Rx_duty2(1:N_highTrunc);
            Rx_duty3 = Rx_duty3(1:N_highTrunc);
        end
        
        %% Distance Shift
        %Rx_signal             = distanceShift(Rx_signal, distance(i), down_part_chirp_rx, fs_high);
        Rx_copy_signal        = distanceShift(Rx_copy_signal, distance(i), down_part_chirp_rx, fs_high);
        Rx_duty1       = distanceShift(Rx_duty1 , distance(i), down_part_chirp_rx, fs_high);
        Rx_duty2       = distanceShift(Rx_duty2 , distance(i), down_part_chirp_rx, fs_high);
        Rx_duty3       = distanceShift(Rx_duty3 , distance(i), down_part_chirp_rx, fs_high);
        % figure
        % subplot(2, 1, 1)
        % pspectrum(Tx_signal, fs_high, 'spectrogram', ...
        %     'TimeResolution', TimeResolu, 'OverlapPercent', 10, 'Leakage', 0.85)
        % title('Tx Signal');
        % subplot(2, 1, 2)
        % pspectrum(Rx_signal, fs_high, 'spectrogram', ...
        %     'TimeResolution', TimeResolu, 'OverlapPercent', 10, 'Leakage', 0.85)
        % title(['Rx signal, distance = ', num2str(distance(i)), ...
        %     ', speed = ', num2str(speed(j))]);
        % savefig(['SignalSpectrum_', num2str(distance(i)), '_', num2str(speed(j)), '.fig'])
        % close(gcf)
        
        %% demodulation --> pre_demo_IF = Rx mix Tx --> IF signal = lowpass(pre_demo_signal)
        %IF_signal = lowpass(Rx_signal .* Tx_signal, freq, fs_high);
        IF_onlyDist  = lowpass(pure_distance_Rx_signal .* Tx_signal, freq, fs_high);
        IF_noRPS  = lowpass(Rx_copy_signal .* Tx_signal, freq, fs_high);
        IF_duty1 = lowpass(Rx_duty1 .* Tx_signal(1:N_highTrunc), freq, fs_high);
        IF_duty2 = lowpass(Rx_duty2 .* Tx_signal(1:N_highTrunc), freq, fs_high);
        IF_duty3 = lowpass(Rx_duty3 .* Tx_signal(1:N_highTrunc), freq, fs_high);
        fft_resolution = 0.5 * fs_high / (N_high);
        fft_truncResolution = 0.5 * fs_high / (N_highTrunc);
        % Time Domain
        % Skipped
        
        % Frequency Domain
        % fft_IF_signal = T_high *abs(fft(IF_signal));
        % fft_IF_noRPS = T_high *abs(fft(IF_noRPS));
        % figure
        % plot(...
        %    f_high(1:0.5 * N_high + 1), fft_IF_signal(1:0.5 * N_high + 1), ':o', ...
        %    f_high(1:0.5 * N_high + 1), fft_IF_noRPS(1:0.5 * N_high + 1), ':o');
        % xlabel('Frequency (Hz)')
        % set(gca, 'YScale', 'log')
        % legend('IF signal', 'IF noRPS');
        % title([...
        %    '(High) IF Spectrum, Freq. Resolution = ', num2str(fft_resolution), ... 
        %    ', distance = ', num2str(distance(i)), ...
        %    ', speed = ', num2str(speed(j))]);
        % xlim([0, 1000])
        % savefig(['IF_', num2str(distance(i)), '_', num2str(speed(j)), '_HR_Spectrum.fig'])
        %¡@close(gcf)
        % Downsample for computation friendly
        IF_onlyDist = IF_onlyDist(1:ds_ratio:numel(IF_onlyDist));
        IF_noRPS  = IF_noRPS(1:ds_ratio:numel(IF_noRPS));
        IF_duty1  = IF_duty1(1:ds_ratio:numel(IF_duty1));
        IF_duty2  = IF_duty2(1:ds_ratio:numel(IF_duty2));
        IF_duty3  = IF_duty3(1:ds_ratio:numel(IF_duty3));
        N_low = numel(IF_noRPS);
        N_lowTrunc = numel(IF_duty1);
        f_lowTrunc = 0:fs_low/N_lowTrunc:fs_low;
        fft_resolution = 0.5 * fs_low / (N_low);
        fft_truncResolution = 0.5 * fs_low / (N_lowTrunc);
        
        %% Figures
        % Time Domain
        figure
        plot(...
            t_low, IF_onlyDist,':o',...
            t_low, IF_noRPS, ':o',...
            t_low(1:N_lowTrunc), IF_duty1, ':o', ...
            t_low(1:N_lowTrunc), IF_duty2, ':o', ...
            t_low(1:N_lowTrunc), IF_duty3, ':o');
        xlabel('Time (s)');
        legend('IF onlyDist','IF Speed (Ideal)','IF Speed (RPS mod 0.3)','IF Speed (RPS mod 0.5)','IF Speed (RPS mod 0.7)');
        title([...
            'IF Signal, distance = ', num2str(distance(i)/100), ...
            ', speed = ', num2str(speed(j)/1000)]);
        savefig(['IF_dist_', num2str(distance(i)/100), '_speed_', num2str(speed(j)/1000), '_Signal.fig'])
        close(gcf)
        
        % Frequency Domain
        fft_IF_onlyDist = T_low *abs(fft(IF_onlyDist));
        fft_IF_noRPS    = T_low *abs(fft(IF_noRPS));
        fft_IF_duty1    = T_low *abs(fft(IF_duty1));
        fft_IF_duty2    = T_low *abs(fft(IF_duty2));
        fft_IF_duty3    = T_low *abs(fft(IF_duty3));
        figure
        plot(...
            f_low(1:0.5 * N_low + 1), fft_IF_onlyDist(1:0.5 * N_low + 1), ':o', ...
            f_low(1:0.5 * N_low + 1), fft_IF_noRPS(1:0.5 * N_low + 1), ':o', ...
            f_lowTrunc(1:0.5 * N_lowTrunc + 1), fft_IF_duty1(1:0.5 * N_lowTrunc + 1), ':x', ...
            f_lowTrunc(1:0.5 * N_lowTrunc + 1), fft_IF_duty2(1:0.5 * N_lowTrunc + 1), ':x', ...
            f_lowTrunc(1:0.5 * N_lowTrunc + 1), fft_IF_duty3(1:0.5 * N_lowTrunc + 1), ':x', 'LineWidth',2, 'MarkerSize',10);
        xlabel('Frequency (Hz)');
        xlim([0, 2000])
        set(gca, 'YScale', 'log')
        legend('IF onlyDist','IF Speed (Ideal)','IF Speed (RPS mod 0.3)','IF Speed (RPS mod 0.5)','IF Speed (RPS mod 0.7)');
        title([...
            '(Low) IF Spectrum, Freq. Res (pre Trunc) = ', num2str(fft_resolution), ...
            ', Freq. Res (post Trunc) = ', num2str(fft_truncResolution), newline,...
            ', distance = ', num2str(distance(i)/100), ...
            ', speed = ', num2str(speed(j)/1000)]);
        savefig(['IF_dist_', num2str(distance(i)/100), '_speed_', num2str(speed(j)/1000), '_Spectrum.fig'])
        close(gcf)
    end
end

function [signal, down_part_chirp] = chirper(freq, fs, t, fm, sweep_num, bw)
    signal = zeros(1, numel(t)); 
    
    %% Part of Cont. Triangular Modulation Signal
    t_up_part   = 0: 1/fs : (0.5 * 1/fm);
    t_down_part = 0: 1/fs : (0.5 * 1/fm);
    up_part_chirp = chirp(t_up_part, freq-0.5*bw, 0.5*(1/fm), freq+0.5*bw);
    down_part_chirp = chirp(t_down_part, freq+0.5*bw, 0.5*(1/fm), freq-0.5*bw);    
    part_chirp = [up_part_chirp(1:(end-1)), down_part_chirp(1:(end-1))];
    
    %% Cont. Triangular Modulation Signal
    part_len = numel(part_chirp);
    for i=0:(sweep_num-1)
        signal((1 + i * part_len): ((i + 1) * part_len)) = part_chirp;
    end
    % filling the remaining part of signal
    signal(sweep_num * part_len + 1:end) = part_chirp(1:(numel(signal) - sweep_num * part_len));
end

% Notes:
% 1. For max(*t*) = 0.1, *speed* should be larger than ~2500.
% 2. Assumed that RCS of the detected object is constant
function Rx_signal = rpsModulation(Rx_signal, distance_shift_signal, t, duty, rps)
    t_cycle = 1 / rps;        
    for i = 1:numel(Rx_signal)
        if rem(t(i), t_cycle) > duty * t_cycle
            Rx_signal(i) = distance_shift_signal(i);
        end
    end
end

% Notes: 
% 1. *start* should be int
% 2. length of *down_part_chirp* is assumed that larger than
%    *time_delay * fs* 
% 3. *signal* is assume to start at chirp signal with
%    instantaneous frequency *center_freq - 0.5 bw*
function signal = distanceShift(signal, distance, down_part_chirp, fs)
    time_delay = 2 * distance / (3e8);
    
    start = numel(down_part_chirp) - fix(time_delay * fs) + 1;
    delay_down_chirp = down_part_chirp(start:end);
    signal = [delay_down_chirp, signal(1:(end - numel(delay_down_chirp)))];
end

