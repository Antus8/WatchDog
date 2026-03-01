clc;
clear all;

%% parameters definition
dt = 0.1; % sample time
e = 10; % noise estimation window size
l = 15; % least squares window size 
Kpos = 30; % number of predicted future points in GT (NOT change it)
n_ts = 15; %number of input trajectory points

filename = pwd +"/evaluation_samples.csv";

%% data loading
disp("loading data from CSV file...");
data = load_data_from_csv(filename);
disp("data loaded");

%% data processing
disp("data processing is starting...")
[input_trajectories, estimated_positions, future_predictions, ground_truths] = run_kalman_ls(data,  dt, e, l, Kpos, n_ts);
disp("data processed")

%% KPI evaluation
disp("KPI computation is starting...")
[ADE, FDE] = compute_ADE_FDE(future_predictions, ground_truths);
disp("KPI computed")

for i=1:length(ADE)
    disp("KPI for t = "+num2str(5*i*dt)+" s:");
    disp(" - Average Displacement Error: "+num2str(ADE{i})+" m");
    disp(" - Final Displacement Error: "+num2str(FDE{i})+" m");
end

%% Plot results
sample_idx = randi(size(input_trajectories,1));
plot_prediction_vs_groundtruth(input_trajectories{sample_idx}, estimated_positions{sample_idx}, ground_truths{sample_idx}, future_predictions{sample_idx}, sample_idx);
