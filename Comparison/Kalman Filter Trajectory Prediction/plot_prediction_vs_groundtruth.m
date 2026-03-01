function plot_prediction_vs_groundtruth(input_traj, estim_traj, gt_future, predicted_future, sample_idx)
    % input_traj: 15x3 (x, y, z)
    % gt_future:  Kposx3 (ground truth futura)
    % predicted_future: 15xKposx3 (predizione al tempo t=1...15), useremo t=1 per confronto

    % Estrai la previsione fatta al primo tempo utile (t=15)
    pred = squeeze(predicted_future(15, :, :));  % Kpos x 3

    % Plot
    figure;
    hold on; grid on; axis equal;
    title(sprintf('Sample #%d - Predizione vs Ground Truth', sample_idx), 'FontWeight', 'bold');
    xlabel('X [m]');
    ylabel('Y [m]');
    zlabel('Z [m]');
    view(3);

    % Measurements
    plot3(input_traj(:,1), input_traj(:,2), input_traj(:,3), '-ok', 'DisplayName', 'Measured trajectory (15t)');
    
    % KF Estimated trajectory
    plot3(estim_traj(:,1), estim_traj(:,2), estim_traj(:,3), '-b', 'LineWidth', 1.5, 'DisplayName', 'KF estimation');

    % Ground truth futura
    plot3(gt_future(:,1), gt_future(:,2), gt_future(:,3), '-g', 'LineWidth', 1.5, 'DisplayName', 'Future Ground Truth');

    % Predizione
    plot3(pred(:,1), pred(:,2), pred(:,3), '-r', 'LineWidth', 1.5, 'DisplayName', 'KF Prediction');

    legend show;
end
