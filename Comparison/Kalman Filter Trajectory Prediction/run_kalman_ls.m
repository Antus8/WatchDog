function [input_trajectories, estimated_positions, future_predictions, ground_truth] = run_kalman_ls(data,  dt, e, l, Kpos, n_timesteps)

    num_samples = size(data, 1) / 2;

    % variables definition
    input_trajectories = cell(num_samples, 1);
    estimated_positions = cell(num_samples, 1);
    estimated_velocities = cell(num_samples, 1);
    future_predictions = cell(num_samples, 1);
    ground_truth = cell(num_samples, 1);

    for i = 1:num_samples
        input_row = data(2*i - 1, 1:n_timesteps*3);
        gt_row = data(2*i, :);

        % Parsing input trajectory (15 timestep)
        input_traj_c = reshape(input_row, [3, n_timesteps])'; % 15x3 (x,y,z)
        input_traj = cell2mat(input_traj_c);

        % Parsing gt trajectory (30 timestep)
        gt_traj_c = reshape(gt_row, [3, Kpos])'; % 30x3 (x,y,z)
        gt_traj = cell2mat(gt_traj_c);


        % Applichiamo il filtro
        %[est_pos, est_vel] = apply_kalman_ls(input_traj, dt, e, l);;
        [est_pos, est_vel, fut_pred] = apply_kalman_ls(input_traj, dt, e, l, Kpos);

        input_trajectories{i} = input_traj;
        estimated_positions{i} = est_pos;
        estimated_velocities{i} = est_vel;
        future_predictions{i} = fut_pred;
        ground_truth{i} = gt_traj;

        % Ground truth è utile per le metriche, ma non viene usato ora
    end

    disp('end of Kalman Filter iterations');
end