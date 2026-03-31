%% run_all_cbf_cases.m
% Wrapper to run multiple CBF cases using main.m and generate comparison plots

clear; close all; clc;

%% -----------------------------
% USER OPTIONS
%% -----------------------------
run_refinement = 0;   % 0 = only initial CBF, 1 = also run cbf_refine
save_figs = 0;        % save figures as png
show_vector_field = 1;

%% -----------------------------
% CASE DEFINITIONS
%% -----------------------------
cases = {
    struct('name','L1_NoInputLimit',  'linear_like_form',1,'include_input_limits',0), ...
    struct('name','L1_WithInputLimit','linear_like_form',1,'include_input_limits',1), ...
    struct('name','L2_NoInputLimit',  'linear_like_form',2,'include_input_limits',0), ...
    struct('name','L2_WithInputLimit','linear_like_form',2,'include_input_limits',1)
};

results = cell(length(cases),1);

%% -----------------------------
% RUN ALL CASES
%% -----------------------------
for k = 1:length(cases)
    fprintf('\n============================================\n');
    fprintf('Running case %d / %d : %s\n', k, length(cases), cases{k}.name);
    fprintf('============================================\n');

    % clear variables except what we need
    clearvars -except cases results k run_refinement save_figs show_vector_field

    % parameters passed into main.m
    linear_like_form = cases{k}.linear_like_form;
    include_input_limits = cases{k}.include_input_limits;
    save_file = 0;
    print_file = 0;
    make_plot = 0;
    x0_user = [];
    
    % run backend
    run('main.m');

    % store result
    cbf_result.case_name = cases{k}.name;
    results{k} = cbf_result;
end

%% -----------------------------
% PLOT 1: ALL GEOMETRIES
%% -----------------------------
figure('Name','All CBF Geometries','Color','w'); hold on; grid on;

colors = lines(length(results));

for k = 1:length(results)
    r = results{k};

    % plot original constraints only once
    if k == 1
        fimplicit(r.c1_fcn, 'r', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.2);
        fimplicit(r.c2_fcn, 'r', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.2);
        fimplicit(r.h0_fcn, 'k', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.5);
    end

    fimplicit(r.h_init_fcn1, [-1.2 1.2 -1.2 1.2], ...
        'Color', colors(k,:), 'LineStyle','--','LineWidth',2);
end

xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
title('Initial CBF Boundaries Across Cases','Interpreter','latex');
axis equal;
xlim([-1.2 1.2]); ylim([-1.2 1.2]);

legend_entries = {'State constraints','Initial guess $h_0(x)=0$'};
for k = 1:length(results)
    legend_entries{end+1} = results{k}.case_name; %#ok<SAGROW>
end

lgd = legend(legend_entries, ...
    'Interpreter','latex', ...
    'Location','southeast');

lgd.Box = 'on';
lgd.FontSize = 10;

if save_figs
    saveas(gcf,'plot_all_geometries.png');
end

%% -----------------------------
% PLOT 2: INPUT LIMIT COMPARISON FOR L1
%% -----------------------------
figure('Name','Input Limit Effect - Linear Like Form 1','Color','w'); hold on; grid on;

r1 = results{1}; % L1 no input limit
r2 = results{2}; % L1 with input limit

fimplicit(r1.c1_fcn, 'r', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.2);
fimplicit(r1.c2_fcn, 'r', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.2);
fimplicit(r1.h0_fcn, 'k', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.5);

fimplicit(r1.h_init_fcn1, [-1.2 1.2 -1.2 1.2], 'b--', 'LineWidth',2);
fimplicit(r2.h_init_fcn1, [-1.2 1.2 -1.2 1.2], 'g--', 'LineWidth',2);

xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
title('Effect of Input Constraints (Linear-like Form 1)','Interpreter','latex');
legend({'$x_1^2=1$','$x_2^2=1$','$h_0(x)=0$','No input limits','With input limits'}, ...
    'Interpreter','latex','Location','best');
axis equal; xlim([-1.2 1.2]); ylim([-1.2 1.2]);

if save_figs
    saveas(gcf,'plot_input_limit_effect_L1.png');
end

%% -----------------------------
% PLOT 3: LINEAR-LIKE FORM COMPARISON WITH INPUT LIMITS
%% -----------------------------
figure('Name','Linear-like Form Comparison','Color','w'); hold on; grid on;

rL1 = results{2}; % L1 with limits
rL2 = results{4}; % L2 with limits

fimplicit(rL1.c1_fcn, 'r', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.2);
fimplicit(rL1.c2_fcn, 'r', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.2);
fimplicit(rL1.h0_fcn, 'k', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.5);

fimplicit(rL1.h_init_fcn1, [-1.2 1.2 -1.2 1.2], 'b--', 'LineWidth',2);
fimplicit(rL2.h_init_fcn1, [-1.2 1.2 -1.2 1.2], 'm--', 'LineWidth',2);

xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
title('Effect of Lifting / Linear-like Form','Interpreter','latex');
legend({'$x_1^2=1$','$x_2^2=1$','$h_0(x)=0$','Form 1','Form 2'}, ...
    'Interpreter','latex','Location','best');
axis equal; xlim([-1.2 1.2]); ylim([-1.2 1.2]);

if save_figs
    saveas(gcf,'plot_linear_like_comparison.png');
end

%% -----------------------------
% PLOT 4: VECTOR FIELD + CBF BOUNDARY
%% -----------------------------
if show_vector_field
    figure('Name','Vector Field with CBF','Color','w'); hold on; grid on;

    r = results{4}; % most interesting case
    [X1, X2] = meshgrid(linspace(-1.1,1.1,25), linspace(-1.1,1.1,25));
    U = X1;
    V = -X1 + 0.5*X2 + X2.^3;

    quiver(X1, X2, U, V, 0.8, 'Color',[0.5 0.5 0.5]);

    fimplicit(r.c1_fcn, 'r', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.2);
    fimplicit(r.c2_fcn, 'r', [-1.2 1.2 -1.2 1.2], 'LineWidth',1.2);
    fimplicit(r.h_init_fcn1, [-1.2 1.2 -1.2 1.2], 'b--', 'LineWidth',2);

    xlabel('$x_1$','Interpreter','latex');
    ylabel('$x_2$','Interpreter','latex');
    title('Open-loop Drift Vector Field with Synthesized CBF Boundary','Interpreter','latex');
    axis equal; xlim([-1.2 1.2]); ylim([-1.2 1.2]);

    if save_figs
        saveas(gcf,'plot_vector_field.png');
    end
end

fprintf('\nDone. All cases executed and plots generated.\n');